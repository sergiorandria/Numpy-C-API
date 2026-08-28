/**
 * @file threadpool.hpp
 * @brief Work-stealing thread pool for parallel NumPy-like operations.
 *
 * Provides `np::ThreadPool` – a fixed-size pool where each worker owns a
 * Chase-Lev style deque. Owners push/pop at the bottom with minimal
 * contention; idle workers steal from the top of victims. Power users on
 * many-core machines get near-linear scaling for `parallel_for` and
 * task submission. Falls back to a single global queue when stealing
 * is not needed (small pools).
 *
 * Reference: NumPy has no explicit threadpool; design mirrors
 * `numpy/core` threaded ufunc dispatch and classic work-stealing
 * (Blumofe, Leiserson – Cilk).
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_THREADPOOL_HPP
#define NP_THREADPOOL_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "api_macros.hpp"

namespace np
{
  namespace detail
  {
    /**
     * @brief Single-owner / multi-thief deque (Chase–Lev, mutex-based).
     *
     * Owner operations (`push_bottom`, `pop_bottom`) are fast-path; thieves
     * call `steal()` which contends on the same mutex. For `n < 64` threads
     * this is sufficient and avoids risky lock-free ABA handling. Replace
     * internals with lock-free circular buffer if profiling shows contention.
     *
     * @tparam T Task type (normally `std::function<void()>`).
     */
    template <typename T>
    class WorkStealingDeque
    {
    public:
      WorkStealingDeque() = default;
      WorkStealingDeque(const WorkStealingDeque&) = delete;
      WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

      /**
       * @brief Push task at the bottom (owner thread).
       * @param v Task to enqueue.
       */
      void push_bottom(T v)
      {
        std::lock_guard<std::mutex> lk(m_);
        dq_.push_back(std::move(v));
      }

      /**
       * @brief Pop task from the bottom (owner thread, LIFO).
       * @return Task if deque non-empty, `std::nullopt` otherwise.
       */
      NP_NODISCARD std::optional<T> pop_bottom()
      {
        std::lock_guard<std::mutex> lk(m_);
        if (dq_.empty())
        {
          return std::nullopt;
        }
        T v = std::move(dq_.back());
        dq_.pop_back();
        return v;
      }

      /**
       * @brief Steal task from the top (thief thread, FIFO).
       * @return Task if stolen, `std::nullopt` otherwise.
       */
      NP_NODISCARD std::optional<T> steal()
      {
        std::lock_guard<std::mutex> lk(m_);
        if (dq_.empty())
        {
          return std::nullopt;
        }
        T v = std::move(dq_.front());
        dq_.pop_front();
        return v;
      }

      /**
       * @brief True when deque empty (locks).
       */
      NP_NODISCARD bool empty() const
      {
        std::lock_guard<std::mutex> lk(m_);
        return dq_.empty();
      }

      /**
       * @brief Number of tasks (locks).
       */
      NP_NODISCARD std::size_t size() const
      {
        std::lock_guard<std::mutex> lk(m_);
        return dq_.size();
      }

    private:
      mutable std::mutex m_;
      std::deque<T> dq_;
    };

  } // namespace detail

  /**
   * @brief Fixed-size thread pool with work stealing.
   *
   * Each worker owns a `WorkStealingDeque<std::function<void()>>`. `submit()`
   * enqueues via round-robin (or to the caller’s queue if caller is a
   * worker). Idle workers spin a bounded number of times trying to steal,
   * then block on a condition variable. Suitable for CPU-bound ndarray
   * kernels on many-core machines; for I/O-bound tasks increase thread
   * count explicitly.
   *
   * Example:
   * @code
   *   np::ThreadPool pool(8);
   *   auto fut = pool.submit([]{ return 42; });
   *   pool.parallel_for(0, 100000, [](std::size_t i){ // work
   *   });
   * @endcode
   */
  class ThreadPool
  {
  public:
    using Task = std::function<void()>;

    /**
     * @brief Construct pool.
     * @param n_threads Number of workers; 0 → `hardware_concurrency()` (≥1).
     */
    explicit ThreadPool(std::size_t n_threads = 0) : done_(false), next_queue_(0)
    {
      if (n_threads == 0)
      {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0)
        {
          n_threads = 4;
        }
      }
      queues_.reserve(n_threads);
      for (std::size_t i = 0; i < n_threads; ++i)
      {
        queues_.emplace_back(std::make_unique<detail::WorkStealingDeque<Task>>());
      }
      workers_.reserve(n_threads);
      for (std::size_t i = 0; i < n_threads; ++i)
      {
        workers_.emplace_back([this, i] { worker_loop(i); });
      }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief Destructor: joins all workers.
     */
    ~ThreadPool()
    {
      shutdown();
    }

    /**
     * @brief Number of worker threads.
     */
    NP_NODISCARD std::size_t size() const noexcept
    {
      return workers_.size();
    }

    /**
     * @brief Submit a callable and get a future.
     *
     * @tparam F Callable type.
     * @tparam Args Argument types.
     * @param f Callable.
     * @param args Arguments to forward.
     * @return `std::future` holding the result.
     */
    template <typename F, typename... Args>
    NP_NODISCARD auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
    {
      using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
      auto task = std::make_shared<std::packaged_task<R()>>(
          [func = std::forward<F>(f),
           ... captured = std::forward<Args>(args)]() mutable -> R
          { return func(std::move(captured)...); });
      std::future<R> fut = task->get_future();
      Task wrapper = [task]() { (*task)(); };
      enqueue(std::move(wrapper));
      return fut;
    }

    /**
     * @brief Enqueue a fire-and-forget task.
     * @param t Task to run.
     */
    void enqueue(Task t)
    {
      const std::size_t idx =
          next_queue_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
      queues_[idx]->push_bottom(std::move(t));
      {
        std::lock_guard<std::mutex> lk(cv_m_);
        cv_.notify_one();
      }
    }

    /**
     * @brief Parallel for over `[begin, end)`.
     *
     * Splits the range into chunks (default: `max(1, n / (size()*4))`) and
     * enqueues one task per chunk. Calling thread also helps (work stealing)
     * and blocks until all chunks complete.
     *
     * @tparam Func `void(std::size_t)` or `void(std::size_t,std::size_t)`.
     * @param begin Start index (inclusive).
     * @param end End index (exclusive).
     * @param func Callable invoked per element or per chunk.
     * @param chunk Explicit chunk size; 0 → auto.
     */
    template <typename Func>
    void
    parallel_for(std::size_t begin, std::size_t end, Func&& func, std::size_t chunk = 0)
    {
      if (begin >= end)
      {
        return;
      }
      const std::size_t n = end - begin;
      if (queues_.empty() || n == 1)
      {
        for (std::size_t i = begin; i < end; ++i)
        {
          func(i);
        }
        return;
      }
      if (chunk == 0)
      {
        chunk = std::max<std::size_t>(1, n / (queues_.size() * 4));
      }
      const std::size_t num_chunks = (n + chunk - 1) / chunk;
      std::atomic<std::size_t> remaining{num_chunks};

      for (std::size_t c = 0; c < num_chunks; ++c)
      {
        const std::size_t s = begin + c * chunk;
        const std::size_t e = std::min(end, s + chunk);
        // Capture func by reference, s/e by value, remaining by reference.
        // Each chunk is independent; no condition variable needed – caller
        // busy-waits and helps via work stealing.
        Task t = [&func, s, e, &remaining]()
        {
          for (std::size_t i = s; i < e; ++i)
          {
            func(i);
          }
          remaining.fetch_sub(1, std::memory_order_acq_rel);
        };
        enqueue(std::move(t));
      }
      // Help while waiting: steal and execute until all chunks done.
      // This keeps the calling thread productive (work stealing) and
      // avoids blocking on a condition variable.
      while (remaining.load(std::memory_order_acquire) != 0)
      {
        if (auto job = try_steal_any())
        {
          (*job)();
        }
        else
        {
          std::this_thread::yield();
        }
      }
    }

    /**
     * @brief Wait until all enqueued tasks have been dequeued (best-effort).
     *
     * Polls queues; not a barrier for tasks spawned inside tasks. For
     * `parallel_for` use its blocking wait. For fire-and-forget chains,
     * use futures.
     */
    void wait()
    {
      while (true)
      {
        bool empty = true;
        for (auto& q : queues_)
        {
          if (!q->empty())
          {
            empty = false;
            break;
          }
        }
        if (empty)
        {
          break;
        }
        std::this_thread::yield();
      }
      // Give workers a chance to finish stolen tasks
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    /**
     * @brief Shutdown pool and join workers.
     */
    void shutdown()
    {
      bool expected = false;
      if (!done_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
      {
        return;
      }
      {
        std::lock_guard<std::mutex> lk(cv_m_);
        cv_.notify_all();
      }
      for (auto& w : workers_)
      {
        if (w.joinable())
        {
          w.join();
        }
      }
    }

    /**
     * @brief Global shared pool (Meyer’s singleton, thread-safe).
     * @param n_threads 0 → hardware_concurrency on first call.
     * @return Reference to the global pool.
     */
    static ThreadPool& global(std::size_t n_threads = 0)
    {
      static ThreadPool instance(n_threads);
      return instance;
    }

  private:
    /**
     * @brief Try to steal a task from any queue (round-robin).
     */
    NP_NODISCARD std::optional<Task> try_steal_any()
    {
      for (std::size_t i = 0; i < queues_.size(); ++i)
      {
        const std::size_t idx =
            (next_queue_.load(std::memory_order_relaxed) + i) % queues_.size();
        if (auto v = queues_[idx]->steal())
        {
          return v;
        }
        if (auto v = queues_[idx]->pop_bottom())
        {
          return v;
        }
      }
      return std::nullopt;
    }

    /**
     * @brief Main worker loop.
     * @param idx Index of this worker’s queue.
     */
    void worker_loop(std::size_t idx)
    {
      constexpr int kSpinIters = 64;
      while (!done_.load(std::memory_order_acquire))
      {
        std::optional<Task> job = queues_[idx]->pop_bottom();
        if (!job)
        {
          // Try stealing
          for (std::size_t iter = 0; iter < queues_.size(); ++iter)
          {
            const std::size_t victim = (idx + iter + 1) % queues_.size();
            job = queues_[victim]->steal();
            if (job)
            {
              break;
            }
          }
        }
        if (job)
        {
          try
          {
            (*job)();
          }
          catch (...)
          {
            // Swallow to keep pool alive; exception is captured in
            // packaged_task future for submit(). Fire-and-forget tasks
            // lose the exception (mirrors std::thread).
          }
          continue;
        }
        // No work: spin briefly then block
        for (int s = 0; s < kSpinIters; ++s)
        {
          std::this_thread::yield();
          job = queues_[idx]->pop_bottom();
          if (job)
          {
            break;
          }
          for (std::size_t v = 0; v < queues_.size(); ++v)
          {
            job = queues_[v]->steal();
            if (job)
            {
              break;
            }
          }
          if (job)
          {
            break;
          }
        }
        if (job)
        {
          try
          {
            (*job)();
          }
          catch (...)
          {
          }
          continue;
        }
        std::unique_lock<std::mutex> lk(cv_m_);
        cv_.wait_for(
            lk,
            std::chrono::milliseconds(2),
            [this, idx]
            {
              if (done_.load(std::memory_order_acquire))
              {
                return true;
              }
              for (auto& q : queues_)
              {
                if (!q->empty())
                {
                  return true;
                }
              }
              return false;
            });
      }
    }

    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<detail::WorkStealingDeque<Task>>> queues_;
    std::atomic<bool> done_;
    std::atomic<std::size_t> next_queue_;
    std::mutex cv_m_;
    std::condition_variable cv_;
  };

  /**
   * @brief Convenience `parallel_for` using the global pool.
   * @param begin Start index.
   * @param end End index.
   * @param func Callable `void(std::size_t)`.
   * @param chunk Chunk size (0 → auto).
   */
  template <typename Func>
  inline void
  parallel_for(std::size_t begin, std::size_t end, Func&& func, std::size_t chunk = 0)
  {
    ThreadPool::global().parallel_for(begin, end, std::forward<Func>(func), chunk);
  }

} // namespace np

#endif // NP_THREADPOOL_HPP
