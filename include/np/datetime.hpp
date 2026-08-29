/**
 * @file datetime.hpp
 * @brief Datetime support (np.datetime_as_string, busday helpers, datetime_data).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.datetime.html
 *
 * Uses std::chrono (C++20) for calendar arithmetic. Dates are
 * represented as `std::chrono::sys_days` (days since 1970-01-01) and
 * stored in `ndarray<int64_t>` as days offset when using the
 * `datetime64[D]` unit (the most common NumPy unit). String arrays
 * use `ndarray<std::string>`.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DATETIME_HPP
#define NP_DATETIME_HPP

#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{
  namespace datetime
  {
    namespace detail
    {
      inline auto broadcast_index_dt(
          const std::vector<int>& in_shape,
          const std::vector<int>& out_shape,
          const std::vector<std::size_t>& out_idx) -> std::vector<std::size_t>
      {
        std::vector<std::size_t> in_idx(in_shape.size(), 0);
        std::size_t out_nd = out_shape.size();
        std::size_t in_nd = in_shape.size();
        for (std::size_t d = 0; d < out_nd; ++d)
        {
          std::ptrdiff_t in_d = static_cast<std::ptrdiff_t>(d)
              - static_cast<std::ptrdiff_t>(out_nd - in_nd);
          if (in_d < 0)
          {
            continue;
          }
          std::size_t id = static_cast<std::size_t>(in_d);
          if (in_shape[id] == 1)
          {
            in_idx[id] = 0;
          }
          else
          {
            in_idx[id] = out_idx[d];
          }
        }
        return in_idx;
      }
    } // namespace detail

    using days = std::chrono::sys_days;
    using sys_days = std::chrono::sys_days;

    /**
     * @brief Business day calendar (np.busdaycalendar).
     *
     * Reference: numpy-reference/reference/generated/numpy.busdaycalendar.html
     */
    struct busdaycalendar
    {
      std::array<bool, 7> weekmask{true, true, true, true, true, false, false};
      std::vector<sys_days> holidays;

      constexpr busdaycalendar() = default;

      explicit busdaycalendar(
          const std::string& weekmask_str, const std::vector<sys_days>& holidays_ = {})
          : holidays(holidays_)
      {
        if (weekmask_str.size() != 7)
        {
          throw std::invalid_argument("busdaycalendar: weekmask must be 7 chars");
        }
        for (std::size_t i = 0; i < 7; ++i)
        {
          weekmask[i] = (weekmask_str[i] == '1');
        }
      }

      explicit busdaycalendar(
          const std::array<bool, 7>& mask, const std::vector<sys_days>& holidays_ = {})
          : weekmask(mask), holidays(holidays_)
      {
      }
    };

    NP_API inline auto _weekday(sys_days d) -> int
    {
      // Monday=0 ... Sunday=6
      std::chrono::weekday wd{d};
      return static_cast<int>(wd.c_encoding() == 7 ? 6 : wd.c_encoding() - 1);
      // Simpler: use std::chrono::weekday::c_encoding gives 0=Sun ...6=Sat
      // Convert: Mon=0 ... Sun=6
    }

    NP_API inline auto _is_holiday(sys_days d, const std::vector<sys_days>& hol) -> bool
    {
      return std::find(hol.begin(), hol.end(), d) != hol.end();
    }

    NP_API inline auto _is_busday(
        sys_days d, const std::array<bool, 7>& weekmask, const std::vector<sys_days>& hol)
        -> bool
    {
      int wd = 0;
      {
        std::chrono::weekday w{d};
        unsigned c = w.c_encoding(); // 0=Sun
        // Map to Mon=0
        wd = (c == 0) ? 6 : static_cast<int>(c) - 1;
      }
      if (!weekmask[static_cast<std::size_t>(wd)])
      {
        return false;
      }
      if (_is_holiday(d, hol))
      {
        return false;
      }
      return true;
    }

    /**
     * @brief Whether dates are business days (np.is_busday).
     *
     * Reference: numpy-reference/reference/generated/numpy.is_busday.html
     */
    NP_API inline auto is_busday(
        const ndarray<sys_days>& dates,
        const std::string& weekmask = "1111100",
        const std::vector<sys_days>& holidays = {},
        const busdaycalendar* busdaycal = nullptr) -> ndarray<bool>
    {
      std::array<bool, 7> mask{};
      std::vector<sys_days> hol = holidays;
      if (busdaycal)
      {
        mask = busdaycal->weekmask;
        hol = busdaycal->holidays;
      }
      else
      {
        if (weekmask.size() != 7)
        {
          throw std::invalid_argument("is_busday: weekmask must be 7 chars");
        }
        for (std::size_t i = 0; i < 7; ++i)
        {
          mask[i] = (weekmask[i] == '1');
        }
      }
      ndarray<bool> out(dates.shape);
      for (std::size_t i = 0; i < dates.size(); ++i)
      {
        sys_days d = dates.data()[dates._flat_logical(i)];
        out.data()[i] = _is_busday(d, mask, hol);
      }
      return out;
    }

    // int64_t days-overload (days since 1970-01-01)
    NP_API inline auto is_busday(
        const ndarray<std::int64_t>& dates,
        const std::string& weekmask = "1111100",
        const std::vector<sys_days>& holidays = {},
        const busdaycalendar* busdaycal = nullptr) -> ndarray<bool>
    {
      ndarray<sys_days> tmp(dates.shape);
      for (std::size_t i = 0; i < dates.size(); ++i)
      {
        int64_t v = dates.data()[dates._flat_logical(i)];
        tmp.data()[i] = sys_days{std::chrono::days{v}};
      }
      return is_busday(tmp, weekmask, holidays, busdaycal);
    }

    /**
     * @brief Offset dates by business days (np.busday_offset).
     *
     * Reference: numpy-reference/reference/generated/numpy.busday_offset.html
     */
    NP_API inline auto busday_offset(
        const ndarray<sys_days>& dates,
        const ndarray<std::int64_t>& offsets,
        const std::string& roll = "raise",
        const std::string& weekmask = "1111100",
        const std::vector<sys_days>& holidays = {},
        const busdaycalendar* busdaycal = nullptr) -> ndarray<sys_days>
    {
      std::array<bool, 7> mask{};
      std::vector<sys_days> hol = holidays;
      if (busdaycal)
      {
        mask = busdaycal->weekmask;
        hol = busdaycal->holidays;
      }
      else
      {
        for (std::size_t i = 0; i < 7; ++i)
        {
          mask[i] = (weekmask[i] == '1');
        }
      }
      std::vector<int> out_shape =
          np::detail::broadcast_shapes(dates.shape, offsets.shape);
      ndarray<sys_days> out(out_shape);
      np::detail::Odometer od(out_shape);
      while (!od.done())
      {
        const auto& idx = od.idx();
        sys_days d = dates.get(detail::broadcast_index_dt(dates.shape, out_shape, idx));
        int64_t off =
            offsets.get(detail::broadcast_index_dt(offsets.shape, out_shape, idx));

        // roll rule
        bool is_bd = _is_busday(d, mask, hol);
        if (!is_bd)
        {
          if (roll == "raise")
          {
            throw std::invalid_argument(
                "busday_offset: date is not a business day and roll='raise'");
          }
          else if (roll == "forward" || roll == "following")
          {
            while (!_is_busday(d, mask, hol))
            {
              d += std::chrono::days{1};
            }
          }
          else if (roll == "backward" || roll == "preceding")
          {
            while (!_is_busday(d, mask, hol))
            {
              d -= std::chrono::days{1};
            }
          }
          else if (roll == "modifiedfollowing")
          {
            sys_days fwd = d;
            while (!_is_busday(fwd, mask, hol))
            {
              fwd += std::chrono::days{1};
            }
            // if next business day is next month, go backward
            auto ymd_orig = std::chrono::year_month_day{d};
            auto ymd_fwd = std::chrono::year_month_day{fwd};
            if (ymd_orig.month() != ymd_fwd.month())
            {
              sys_days bwd = d;
              while (!_is_busday(bwd, mask, hol))
              {
                bwd -= std::chrono::days{1};
              }
              d = bwd;
            }
            else
            {
              d = fwd;
            }
          }
          else if (roll == "modifiedpreceding")
          {
            sys_days bwd = d;
            while (!_is_busday(bwd, mask, hol))
            {
              bwd -= std::chrono::days{1};
            }
            auto ymd_orig = std::chrono::year_month_day{d};
            auto ymd_bwd = std::chrono::year_month_day{bwd};
            if (ymd_orig.month() != ymd_bwd.month())
            {
              sys_days fwd = d;
              while (!_is_busday(fwd, mask, hol))
              {
                fwd += std::chrono::days{1};
              }
              d = fwd;
            }
            else
            {
              d = bwd;
            }
          }
          else if (roll == "nat")
          {
            // remain NaT – we throw as not representable; use sentinel min
            throw std::invalid_argument(
                "busday_offset: NaT not supported, use raise/forward");
          }
          else
          {
            throw std::invalid_argument("busday_offset: unknown roll '" + roll + "'");
          }
        }

        // Apply offset in business days
        if (off > 0)
        {
          for (int64_t k = 0; k < off; ++k)
          {
            do
            {
              d += std::chrono::days{1};
            } while (!_is_busday(d, mask, hol));
          }
        }
        else if (off < 0)
        {
          for (int64_t k = 0; k < -off; ++k)
          {
            do
            {
              d -= std::chrono::days{1};
            } while (!_is_busday(d, mask, hol));
          }
        }
        out.set(idx, d);
        od.advance();
      }
      return out;
    }

    /**
     * @brief Count business days between dates (np.busday_count).
     *
     * Reference: numpy-reference/reference/generated/numpy.busday_count.html
     */
    NP_API inline auto busday_count(
        const ndarray<sys_days>& begindates,
        const ndarray<sys_days>& enddates,
        const std::string& weekmask = "1111100",
        const std::vector<sys_days>& holidays = {},
        const busdaycalendar* busdaycal = nullptr) -> ndarray<std::int64_t>
    {
      std::array<bool, 7> mask{};
      std::vector<sys_days> hol = holidays;
      if (busdaycal)
      {
        mask = busdaycal->weekmask;
        hol = busdaycal->holidays;
      }
      else
      {
        for (std::size_t i = 0; i < 7; ++i)
        {
          mask[i] = (weekmask[i] == '1');
        }
      }
      std::vector<int> out_shape =
          np::detail::broadcast_shapes(begindates.shape, enddates.shape);
      ndarray<std::int64_t> out(out_shape);
      np::detail::Odometer od(out_shape);
      while (!od.done())
      {
        const auto& idx = od.idx();
        sys_days b =
            begindates.get(detail::broadcast_index_dt(begindates.shape, out_shape, idx));
        sys_days e =
            enddates.get(detail::broadcast_index_dt(enddates.shape, out_shape, idx));
        int64_t cnt = 0;
        sys_days cur = b;
        if (cur <= e)
        {
          while (cur < e)
          {
            if (_is_busday(cur, mask, hol))
            {
              ++cnt;
            }
            cur += std::chrono::days{1};
          }
        }
        else
        {
          while (cur > e)
          {
            cur -= std::chrono::days{1};
            if (_is_busday(cur, mask, hol))
            {
              --cnt;
            }
          }
        }
        out.set(idx, cnt);
        od.advance();
      }
      return out;
    }

    /**
     * @brief Convert datetime array to string array (np.datetime_as_string).
     *
     * Reference: numpy-reference/reference/generated/numpy.datetime_as_string.html
     */
    NP_API inline auto
    datetime_as_string(const ndarray<sys_days>& arr, const std::string& unit = "D")
        -> ndarray<std::string>
    {
      (void)unit;
      ndarray<std::string> out(arr.shape);
      for (std::size_t i = 0; i < arr.size(); ++i)
      {
        sys_days d = arr.data()[arr._flat_logical(i)];
        std::chrono::year_month_day ymd{d};
        int y = static_cast<int>(ymd.year());
        unsigned m = static_cast<unsigned>(ymd.month());
        unsigned dd = static_cast<unsigned>(ymd.day());
        char buf[11];
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", y, m, dd);
        out.data()[i] = std::string(buf);
      }
      return out;
    }

    NP_API inline auto
    datetime_as_string(const ndarray<std::int64_t>& arr, const std::string& unit = "D")
        -> ndarray<std::string>
    {
      ndarray<sys_days> tmp(arr.shape);
      for (std::size_t i = 0; i < arr.size(); ++i)
      {
        tmp.data()[i] = sys_days{std::chrono::days{arr.data()[arr._flat_logical(i)]}};
      }
      return datetime_as_string(tmp, unit);
    }

    /**
     * @brief Return (unit, count) for dtype (np.datetime_data).
     *
     * Reference: numpy-reference/reference/generated/numpy.datetime_data.html
     */
    NP_API inline auto datetime_data(const std::string& dtype_str)
        -> std::pair<std::string, int>
    {
      // Parse strings like "datetime64[D]", "timedelta64[ms]"
      auto l = dtype_str.find('[');
      auto r = dtype_str.find(']');
      if (l == std::string::npos || r == std::string::npos || r <= l + 1)
      {
        return {"generic", 1};
      }
      std::string unit = dtype_str.substr(l + 1, r - l - 1);
      return {unit, 1};
    }

    // ── NaT / isnat / scalar type aliases (np.datetime64 / np.timedelta64)
    /**
     * @brief NaT sentinel (Not a Time).
     *
     * NumPy uses the minimum int64 datetime64 value as NaT. We mirror it
     * with `sys_days::min()` so `sys_days(NaT) == min` is detectable.
     *
     * Reference: numpy-reference/reference/generated/numpy.isnat.html
     */
    inline constexpr sys_days NaT = sys_days::min();

    using datetime64 = sys_days;
    using timedelta64 = std::chrono::days;

    /**
     * @brief Test for NaT (np.isnat).
     *
     * Reference: numpy-reference/reference/generated/numpy.isnat.html
     */
    NP_API inline auto isnat(const ndarray<sys_days>& arr) -> ndarray<bool>
    {
      ndarray<bool> out(arr.shape);
      for (std::size_t i = 0; i < arr.size(); ++i)
      {
        out.data()[i] = (arr.data()[arr._flat_logical(i)] == NaT);
      }
      return out;
    }

    NP_API inline auto isnat(const ndarray<std::int64_t>& arr) -> ndarray<bool>
    {
      ndarray<bool> out(arr.shape);
      constexpr std::int64_t kNaT = std::numeric_limits<std::int64_t>::min();
      for (std::size_t i = 0; i < arr.size(); ++i)
      {
        out.data()[i] = (arr.data()[arr._flat_logical(i)] == kNaT);
      }
      return out;
    }

    NP_API inline auto isnat(sys_days v) -> bool
    {
      return v == NaT;
    }

    NP_API inline auto isnat(std::int64_t v) -> bool
    {
      return v == std::numeric_limits<std::int64_t>::min();
    }

  } // namespace datetime

  // Top-level mirrors – `np::isnat` / `np::NaT` match `numpy.*`
  // `datetime64` / `timedelta64` are kept inside `np::datetime` to avoid
  // collision with `np::datetime64`/`timedelta64` dtype tags in `dtype.hpp`.
  using datetime::isnat;
  using datetime::NaT;
} // namespace np

#endif // NP_DATETIME_HPP
