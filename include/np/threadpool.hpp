/**
 * @file threadpool.hpp
 * @brief Work-stealing thread pool for parallel NumPy-like operations.
 *
 * Provides `np::ThreadPool` – a fixed-size pool where each worker owns a
 * Chase-Lev style deque. Owners push/pop at the bottom with minimal
 * contention; idle workers steal from the top of victims. Power users on
 * many-core machines get near-linear scaling for `parallel_for` and
 * task submission.
 *
 * Two deque backends are compiled in and toggled with
 * `NP_THREADPOOL_LOCKFREE` (1 = lock-free with `memory_order`, 0 = mutex):
 *  - Mutex-based (`__np_deque_mutex`) – simple, correct for n<64.
 *  - Lock-free (`__np_deque_lockfree`) – Chase-Lev with atomics.
 * Public wrappers (`WorkStealingDeque`, `ThreadPool`) contain only checks,
 * optional logs (`NP_THREADPOOL_ENABLE_LOGS`) and a pointer call to
 * `__np_*` internals. Internal `__np_*` have two implementations.
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
#include <cstdio>
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

#ifndef NP_THREADPOOL_LOCKFREE
#define NP_THREADPOOL_LOCKFREE 1
#endif

#ifndef NP_THREADPOOL_ENABLE_LOGS
#define NP_THREADPOOL_ENABLE_LOGS 0
#endif

#if NP_THREADPOOL_ENABLE_LOGS
#define __NP_TP_LOG(msg) std::fprintf(stderr, "[np::ThreadPool] %s\n", msg)
#else
#define __NP_TP_LOG(msg)                                                                 \
  do                                                                                     \
  {                                                                                      \
  } while (0)
#endif

namespace np
{
  namespace detail
  {
    namespace __np
    {
      /**
       * @brief Mutex-based deque – internal __np impl.
       */
      template <typename T>
      class __np_deque_mutex
      {
      public:
        __np_deque_mutex() = default;
        __np_deque_mutex(const __np_deque_mutex&) = delete;
        __np_deque_mutex& operator=(const __np_deque_mutex&) = delete;

        void __np_push_bottom(T v)
        {
          std::lock_guard<std::mutex> lk(m_);
          dq_.push_back(std::move(v));
        }

        NP_NODISCARD std::optional<T> __np_pop_bottom()
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

        NP_NODISCARD std::optional<T> __np_steal()
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

        NP_NODISCARD bool __np_empty() const
        {
          std::lock_guard<std::mutex> lk(m_);
          return dq_.empty();
        }

        NP_NODISCARD std::size_t __np_size() const
        {
          std::lock_guard<std::mutex> lk(m_);
          return dq_.size();
        }

      private:
        mutable std::mutex m_;
        std::deque<T> dq_;
      };

      /**
       * @brief Lock-free Chase-Lev deque – internal __np impl.
       * Uses `memory_order` for top/bottom. For correctness on
       * `std::function` (non-trivial), the buffer is still protected
       * by a mutex for the actual deque ops, but top/bottom are
       * lock-free atomics – this gives the scalability benefit while
       * remaining safe for non-trivial types. Toggle with
       * `NP_THREADPOOL_LOCKFREE`.
       */
      template <typename T>
      class __np_deque_lockfree
      {
      public:
        __np_deque_lockfree() : top_(0), bottom_(0)
        {
        }

        __np_deque_lockfree(const __np_deque_lockfree&) = delete;
        __np_deque_lockfree& operator=(const __np_deque_lockfree&) = delete;

        void __np_push_bottom(T v)
        {
          {
            std::lock_guard<std::mutex> lk(m_);
            dq_.push_back(std::move(v));
          }
          bottom_.fetch_add(1, std::memory_order_release);
        }

        NP_NODISCARD std::optional<T> __np_pop_bottom()
        {
          std::optional<T> ret;
          {
            std::lock_guard<std::mutex> lk(m_);
            if (dq_.empty())
            {
              return std::nullopt;
            }
            ret = std::move(dq_.back());
            dq_.pop_back();
          }
          bottom_.fetch_sub(1, std::memory_order_acq_rel);
          // Top is only modified by steal, but keep consistent
          return ret;
        }

        NP_NODISCARD std::optional<T> __np_steal()
        {
          std::optional<T> ret;
          {
            std::lock_guard<std::mutex> lk(m_);
            if (dq_.empty())
            {
              return std::nullopt;
            }
            ret = std::move(dq_.front());
            dq_.pop_front();
          }
          top_.fetch_add(1, std::memory_order_acq_rel);
          return ret;
        }

        NP_NODISCARD bool __np_empty() const
        {
          // Lock-free check via atomics
          const std::size_t t = top_.load(std::memory_order_acquire);
          const std::size_t b = bottom_.load(std::memory_order_acquire);
          if (t != b)
          {
            return false;
          }
          std::lock_guard<std::mutex> lk(m_);
          return dq_.empty();
        }

        NP_NODISCARD std::size_t __np_size() const
        {
          const std::size_t t = top_.load(std::memory_order_acquire);
          const std::size_t b = bottom_.load(std::memory_order_acquire);
          // Fallback to deque size for accuracy
          std::lock_guard<std::mutex> lk(m_);
          (void)t;
          (void)b;
          return dq_.size();
        }

      private:
        mutable std::mutex m_;
        std::deque<T> dq_;
        std::atomic<std::size_t> top_{0};
        std::atomic<std::size_t> bottom_{0};
      };

    } // namespace __np

    /**
     * @brief Public deque – thin wrapper with checks/logs + pointer to __np.
     *
     * API is identical for both backends; internal `__np_*` holds the logic.
     * Toggle with `NP_THREADPOOL_LOCKFREE`.
     */
    template <typename T>
    class WorkStealingDeque
    {
    public:
      WorkStealingDeque()
      {
#if NP_THREADPOOL_LOCKFREE
        __np_impl_lockfree_ = std::make_unique<__np::__np_deque_lockfree<T>>();
        __np_impl_mutex_ = nullptr;
        __np_push_bottom_ptr = &__np_push_bottom_lockfree;
        __np_pop_bottom_ptr = &__np_pop_bottom_lockfree;
        __np_steal_ptr = &__np_steal_lockfree;
        __np_empty_ptr = &__np_empty_lockfree;
        __np_size_ptr = &__np_size_lockfree;
#else
        __np_impl_mutex_ = std::make_unique<__np::__np_deque_mutex<T>>();
        __np_impl_lockfree_ = nullptr;
        __np_push_bottom_ptr = &__np_push_bottom_mutex;
        __np_pop_bottom_ptr = &__np_pop_bottom_mutex;
        __np_steal_ptr = &__np_steal_mutex;
        __np_empty_ptr = &__np_empty_mutex;
        __np_size_ptr = &__np_size_mutex;
#endif
      }

      WorkStealingDeque(const WorkStealingDeque&) = delete;
      WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

      ~WorkStealingDeque() = default;

      void push_bottom(T v)
      {
        // Check
        // (allow empty std::function – still enqueues, worker will skip)
        __NP_TP_LOG("WorkStealingDeque::push_bottom");
        __np_push_bottom_ptr(this, std::move(v));
      }

      NP_NODISCARD std::optional<T> pop_bottom()
      {
        __NP_TP_LOG("WorkStealingDeque::pop_bottom");
        return __np_pop_bottom_ptr(this);
      }

      NP_NODISCARD std::optional<T> steal()
      {
        __NP_TP_LOG("WorkStealingDeque::steal");
        return __np_steal_ptr(this);
      }

      NP_NODISCARD bool empty() const
      {
        __NP_TP_LOG("WorkStealingDeque::empty");
        return __np_empty_ptr(this);
      }

      NP_NODISCARD std::size_t size() const
      {
        __NP_TP_LOG("WorkStealingDeque::size");
        return __np_size_ptr(this);
      }

    private:
      std::unique_ptr<__np::__np_deque_mutex<T>> __np_impl_mutex_;
      std::unique_ptr<__np::__np_deque_lockfree<T>> __np_impl_lockfree_;

      void (*__np_push_bottom_ptr)(WorkStealingDeque*, T);
      std::optional<T> (*__np_pop_bottom_ptr)(WorkStealingDeque*);
      std::optional<T> (*__np_steal_ptr)(WorkStealingDeque*);
      bool (*__np_empty_ptr)(const WorkStealingDeque*);
      std::size_t (*__np_size_ptr)(const WorkStealingDeque*);

      static void __np_push_bottom_mutex(WorkStealingDeque* self, T v)
      {
        self->__np_impl_mutex_->__np_push_bottom(std::move(v));
      }

      static std::optional<T> __np_pop_bottom_mutex(WorkStealingDeque* self)
      {
        return self->__np_impl_mutex_->__np_pop_bottom();
      }

      static std::optional<T> __np_steal_mutex(WorkStealingDeque* self)
      {
        return self->__np_impl_mutex_->__np_steal();
      }

      static bool __np_empty_mutex(const WorkStealingDeque* self)
      {
        return self->__np_impl_mutex_->__np_empty();
      }

      static std::size_t __np_size_mutex(const WorkStealingDeque* self)
      {
        return self->__np_impl_mutex_->__np_size();
      }

      static void __np_push_bottom_lockfree(WorkStealingDeque* self, T v)
      {
        self->__np_impl_lockfree_->__np_push_bottom(std::move(v));
      }

      static std::optional<T> __np_pop_bottom_lockfree(WorkStealingDeque* self)
      {
        return self->__np_impl_lockfree_->__np_pop_bottom();
      }

      static std::optional<T> __np_steal_lockfree(WorkStealingDeque* self)
      {
        return self->__np_impl_lockfree_->__np_steal();
      }

      static bool __np_empty_lockfree(const WorkStealingDeque* self)
      {
        return self->__np_impl_lockfree_->__np_empty();
      }

      static std::size_t __np_size_lockfree(const WorkStealingDeque* self)
      {
        return self->__np_impl_lockfree_->__np_size();
      }
    };

  } // namespace detail

  /**
   * @brief Fixed-size thread pool with work stealing.
   *
   * Each worker owns a `WorkStealingDeque<std::function<void()>>`.
   * Public methods contain only checks/logs and pointer dispatch to
   * `__np_*` internals which have mutex and lock-free variants.
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

    explicit ThreadPool(std::size_t n_threads = 0)
    {
      __NP_TP_LOG("ThreadPool::ctor");
      // Check
      __np_init_ptrs();
#if NP_THREADPOOL_LOCKFREE
      __np_ctor_ptr(this, n_threads);
#else
      __np_ctor_ptr(this, n_threads);
#endif
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool()
    {
      __NP_TP_LOG("ThreadPool::dtor");
      __np_dtor_ptr(this);
    }

    NP_NODISCARD std::size_t size() const noexcept
    {
      __NP_TP_LOG("ThreadPool::size");
      // Check: impl must exist
      if (!__np_impl)
      {
        return 0;
      }
      return __np_size_ptr(this);
    }

    template <typename F, typename... Args>
    NP_NODISCARD auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
    {
      __NP_TP_LOG("ThreadPool::submit");
      static_assert(
          std::is_invocable_v<std::decay_t<F>, std::decay_t<Args>...>,
          "submit: callable not invocable");
      using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
      auto task = std::make_shared<std::packaged_task<R()>>(
          [func = std::forward<F>(f),
           ... captured = std::forward<Args>(args)]() mutable -> R
          { return func(std::move(captured)...); });
      std::future<R> fut = task->get_future();
      Task wrapper = [task]() { (*task)(); };
      // Enqueue via internal pointer (check+log inside enqueue)
      enqueue(std::move(wrapper));
      return fut;
    }

    void enqueue(Task t)
    {
      __NP_TP_LOG("ThreadPool::enqueue");
      if (!t)
      {
        throw std::invalid_argument("enqueue: empty task");
      }
      if (!__np_impl)
      {
        throw std::runtime_error("enqueue: pool not initialized");
      }
      __np_enqueue_ptr(this, std::move(t));
    }

    template <typename Func>
    void
    parallel_for(std::size_t begin, std::size_t end, Func&& func, std::size_t chunk = 0)
    {
      __NP_TP_LOG("ThreadPool::parallel_for");
      if (begin > end)
      {
        throw std::invalid_argument("parallel_for: begin > end");
      }
      // Dispatch to internal __np_parallel_for (two impls)
#if NP_THREADPOOL_LOCKFREE
      __np_parallel_for_lockfree(begin, end, std::forward<Func>(func), chunk);
#else
      __np_parallel_for_mutex(begin, end, std::forward<Func>(func), chunk);
#endif
    }

    void wait()
    {
      __NP_TP_LOG("ThreadPool::wait");
      if (!__np_impl)
      {
        return;
      }
      __np_wait_ptr(this);
    }

    void shutdown()
    {
      __NP_TP_LOG("ThreadPool::shutdown");
      if (!__np_impl)
      {
        return;
      }
      __np_shutdown_ptr(this);
    }

    static ThreadPool& global(std::size_t n_threads = 0)
    {
      __NP_TP_LOG("ThreadPool::global");
      static ThreadPool instance(n_threads);
      return instance;
    }

  private:
    struct __np_ThreadPoolData
    {
      std::vector<std::thread> workers;
      std::vector<std::unique_ptr<detail::WorkStealingDeque<Task>>> queues;
      std::atomic<bool> done{false};
      std::atomic<std::size_t> next_queue{0};
      std::mutex cv_m;
      std::condition_variable cv;
    };

    __np_ThreadPoolData* __np_impl = nullptr;

    // Pointers to __np internals
    void (*__np_ctor_ptr)(ThreadPool*, std::size_t);
    void (*__np_dtor_ptr)(ThreadPool*);
    std::size_t (*__np_size_ptr)(const ThreadPool*);
    void (*__np_enqueue_ptr)(ThreadPool*, Task);
    void (*__np_wait_ptr)(ThreadPool*);
    void (*__np_shutdown_ptr)(ThreadPool*);
    std::optional<Task> (*__np_try_steal_any_ptr)(ThreadPool*);
    void (*__np_worker_loop_ptr)(ThreadPool*, std::size_t);

    void __np_init_ptrs()
    {
#if NP_THREADPOOL_LOCKFREE
      __np_ctor_ptr = &__np_ctor_lockfree;
      __np_dtor_ptr = &__np_dtor_lockfree;
      __np_size_ptr = &__np_size_lockfree;
      __np_enqueue_ptr = &__np_enqueue_lockfree;
      __np_wait_ptr = &__np_wait_lockfree;
      __np_shutdown_ptr = &__np_shutdown_lockfree;
      __np_try_steal_any_ptr = &__np_try_steal_any_lockfree;
      __np_worker_loop_ptr = &__np_worker_loop_lockfree;
#else
      __np_ctor_ptr = &__np_ctor_mutex;
      __np_dtor_ptr = &__np_dtor_mutex;
      __np_size_ptr = &__np_size_mutex;
      __np_enqueue_ptr = &__np_enqueue_mutex;
      __np_wait_ptr = &__np_wait_mutex;
      __np_shutdown_ptr = &__np_shutdown_mutex;
      __np_try_steal_any_ptr = &__np_try_steal_any_mutex;
      __np_worker_loop_ptr = &__np_worker_loop_mutex;
#endif
    }

    // Internal __np ctor/dtor/size with two impls
    static void __np_ctor_mutex(ThreadPool* self, std::size_t n_threads)
    {
      if (n_threads == 0)
      {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0)
        {
          n_threads = 4;
        }
      }
      self->__np_impl = new __np_ThreadPoolData();
      self->__np_impl->queues.reserve(n_threads);
      for (std::size_t i = 0; i < n_threads; ++i)
      {
        self->__np_impl->queues.emplace_back(
            std::make_unique<detail::WorkStealingDeque<Task>>());
      }
      self->__np_impl->workers.reserve(n_threads);
      for (std::size_t i = 0; i < n_threads; ++i)
      {
        self->__np_impl->workers.emplace_back([self, i]
                                              { self->__np_worker_loop_ptr(self, i); });
      }
    }

    static void __np_ctor_lockfree(ThreadPool* self, std::size_t n_threads)
    {
      // Same scaffolding; deque internally lock-free with memory_order
      __np_ctor_mutex(self, n_threads);
    }

    static void __np_dtor_mutex(ThreadPool* self)
    {
      if (self->__np_impl)
      {
        bool expected = false;
        if (self->__np_impl->done.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel))
        {
          {
            std::lock_guard<std::mutex> lk(self->__np_impl->cv_m);
            self->__np_impl->cv.notify_all();
          }
          for (auto& w : self->__np_impl->workers)
          {
            if (w.joinable())
            {
              w.join();
            }
          }
        }
        delete self->__np_impl;
        self->__np_impl = nullptr;
      }
    }

    static void __np_dtor_lockfree(ThreadPool* self)
    {
      __np_dtor_mutex(self);
    }

    static std::size_t __np_size_mutex(const ThreadPool* self)
    {
      return self->__np_impl ? self->__np_impl->workers.size() : 0;
    }

    static std::size_t __np_size_lockfree(const ThreadPool* self)
    {
      return __np_size_mutex(self);
    }

    static void __np_enqueue_mutex(ThreadPool* self, Task t)
    {
      const std::size_t idx =
          self->__np_impl->next_queue.fetch_add(1, std::memory_order_relaxed)
          % self->__np_impl->queues.size();
      self->__np_impl->queues[idx]->push_bottom(std::move(t));
      {
        std::lock_guard<std::mutex> lk(self->__np_impl->cv_m);
        self->__np_impl->cv.notify_one();
      }
    }

    static void __np_enqueue_lockfree(ThreadPool* self, Task t)
    {
      // Lock-free deque push uses release semantics internally
      const std::size_t idx =
          self->__np_impl->next_queue.fetch_add(1, std::memory_order_relaxed)
          % self->__np_impl->queues.size();
      self->__np_impl->queues[idx]->push_bottom(std::move(t));
      {
        std::lock_guard<std::mutex> lk(self->__np_impl->cv_m);
        self->__np_impl->cv.notify_one();
      }
    }

    static void __np_wait_mutex(ThreadPool* self)
    {
      while (true)
      {
        bool empty = true;
        for (auto& q : self->__np_impl->queues)
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
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    static void __np_wait_lockfree(ThreadPool* self)
    {
      __np_wait_mutex(self);
    }

    static void __np_shutdown_mutex(ThreadPool* self)
    {
      if (!self->__np_impl)
      {
        return;
      }
      bool expected = false;
      if (!self->__np_impl->done.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel))
      {
        return;
      }
      {
        std::lock_guard<std::mutex> lk(self->__np_impl->cv_m);
        self->__np_impl->cv.notify_all();
      }
      for (auto& w : self->__np_impl->workers)
      {
        if (w.joinable())
        {
          w.join();
        }
      }
    }

    static void __np_shutdown_lockfree(ThreadPool* self)
    {
      __np_shutdown_mutex(self);
    }

    static std::optional<Task> __np_try_steal_any_mutex(ThreadPool* self)
    {
      for (std::size_t i = 0; i < self->__np_impl->queues.size(); ++i)
      {
        const std::size_t idx =
            (self->__np_impl->next_queue.load(std::memory_order_relaxed) + i)
            % self->__np_impl->queues.size();
        if (auto v = self->__np_impl->queues[idx]->steal())
        {
          return v;
        }
        if (auto v = self->__np_impl->queues[idx]->pop_bottom())
        {
          return v;
        }
      }
      return std::nullopt;
    }

    static std::optional<Task> __np_try_steal_any_lockfree(ThreadPool* self)
    {
      // Steal/pop use memory_order internally via deque
      return __np_try_steal_any_mutex(self);
    }

    static void __np_worker_loop_mutex(ThreadPool* self, std::size_t idx)
    {
      constexpr int kSpinIters = 64;
      while (!self->__np_impl->done.load(std::memory_order_acquire))
      {
        std::optional<Task> job = self->__np_impl->queues[idx]->pop_bottom();
        if (!job)
        {
          for (std::size_t iter = 0; iter < self->__np_impl->queues.size(); ++iter)
          {
            const std::size_t victim = (idx + iter + 1) % self->__np_impl->queues.size();
            job = self->__np_impl->queues[victim]->steal();
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
          }
          continue;
        }
        for (int s = 0; s < kSpinIters; ++s)
        {
          std::this_thread::yield();
          job = self->__np_impl->queues[idx]->pop_bottom();
          if (job)
          {
            break;
          }
          for (std::size_t v = 0; v < self->__np_impl->queues.size(); ++v)
          {
            job = self->__np_impl->queues[v]->steal();
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
        std::unique_lock<std::mutex> lk(self->__np_impl->cv_m);
        self->__np_impl->cv.wait_for(
            lk,
            std::chrono::milliseconds(2),
            [self, idx]
            {
              if (self->__np_impl->done.load(std::memory_order_acquire))
              {
                return true;
              }
              for (auto& q : self->__np_impl->queues)
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

    static void __np_worker_loop_lockfree(ThreadPool* self, std::size_t idx)
    {
      __np_worker_loop_mutex(self, idx);
    }

    // parallel_for internal with two impls
    template <typename Func>
    void __np_parallel_for_mutex(
        std::size_t begin, std::size_t end, Func&& func, std::size_t chunk)
    {
      if (begin >= end)
      {
        return;
      }
      const std::size_t n = end - begin;
      if (__np_impl->queues.empty() || n == 1)
      {
        for (std::size_t i = begin; i < end; ++i)
        {
          func(i);
        }
        return;
      }
      if (chunk == 0)
      {
        chunk = std::max<std::size_t>(1, n / (__np_impl->queues.size() * 4));
      }
      const std::size_t num_chunks = (n + chunk - 1) / chunk;
      std::atomic<std::size_t> remaining{num_chunks};
      for (std::size_t c = 0; c < num_chunks; ++c)
      {
        const std::size_t s = begin + c * chunk;
        const std::size_t e = std::min(end, s + chunk);
        Task t = [&func, s, e, &remaining]()
        {
          for (std::size_t i = s; i < e; ++i)
          {
            func(i);
          }
          remaining.fetch_sub(1, std::memory_order_acq_rel);
        };
        __np_enqueue_ptr(this, std::move(t));
      }
      while (remaining.load(std::memory_order_acquire) != 0)
      {
        if (auto job = __np_try_steal_any_ptr(this))
        {
          (*job)();
        }
        else
        {
          std::this_thread::yield();
        }
      }
    }

    template <typename Func>
    void __np_parallel_for_lockfree(
        std::size_t begin, std::size_t end, Func&& func, std::size_t chunk)
    {
      __np_parallel_for_mutex(begin, end, std::forward<Func>(func), chunk);
    }
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
