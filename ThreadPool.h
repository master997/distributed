#pragma once

// Thread pool implementation - the class shown in DC-8 lecture slides 6-10.
// A fixed group of worker threads sits asleep on a condition variable. When
// a task is enqueued, ONE worker is woken up, runs the task, then goes back
// to sleep. The pool is destroyed at program exit (the destructor wakes all
// workers up so they can exit cleanly).
//
// Why we want this here: spawning a fresh std::thread costs roughly 30-100
// microseconds on macOS. Across 10 timed iterations of matrixOperationsInit,
// each spawning ~24 threads (8 per operation, 3 operations), that's about
// 250 spawns. By keeping the workers alive between calls we pay the spawn
// cost once per program, not once per task.

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <memory>
#include <stdexcept>

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false)
    {
        // Spin up the worker threads. Each one sits in a loop pulling tasks
        // off the shared queue (this is the producer-consumer pattern from
        // DC-7 lecture, with the dispatcher acting as producer and the
        // workers as consumers).
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        // Take the queue lock so we can safely read or wait.
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        // Sleep until either there's a task or the pool is
                        // being destroyed. The CV wait releases the mutex
                        // while sleeping (DC-6 lecture).
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        // If the pool is being shut down and there's
                        // nothing left to do, exit the worker.
                        if (this->stop && this->tasks.empty()) return;
                        // Move the task out of the queue under the lock.
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    // Lock released - actually run the task.
                    task();
                }
            });
        }
    }

    // Submit a task. Accepts any callable + args, returns a std::future for
    // the result so the caller can wait on it. Same template signature as
    // the one in DC-8 lecture slide 6.
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        // Pack the call into a packaged_task so we get a future for free.
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            // Wrap the packaged_task in a void() lambda so the queue can
            // hold a uniform type regardless of return type.
            tasks.emplace([task]() { (*task)(); });
        }
        // Wake exactly one worker - no point waking the rest if there's
        // only one new task waiting.
        condition.notify_one();
        return res;
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        // Wake everyone so they can see stop=true and exit their loop.
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }

    // Pools shouldn't be copyable - copying would mean two owners trying to
    // join the same threads.
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    std::vector<std::thread>          workers;
    std::queue<std::function<void()>> tasks;
    std::mutex                        queue_mutex;
    std::condition_variable           condition;
    bool                              stop;
};
