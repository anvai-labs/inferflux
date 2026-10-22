#include "io/async_file_writer.h"

#include "runtime/thread_name.h"

#include <fstream>

namespace inferflux {

AsyncFileWriter::AsyncFileWriter(std::size_t max_queue_depth)
    : max_queue_depth_(max_queue_depth) {}

AsyncFileWriter::~AsyncFileWriter() { Stop(); }

void AsyncFileWriter::Configure(std::size_t workers,
                                std::size_t max_queue_depth) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (workers == 0) {
    workers = 1;
  }
  preferred_workers_ = workers;
  if (max_queue_depth > 0) {
    max_queue_depth_ = max_queue_depth;
    producer_cv_.notify_all();
  }
}

void AsyncFileWriter::Start(std::size_t workers) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
      return;
    }
    stop_ = false;
    running_ = true;
    if (workers == 0) {
      workers = preferred_workers_;
    }
    preferred_workers_ = workers == 0 ? 1 : workers;
  }
  for (std::size_t i = 0; i < preferred_workers_; ++i) {
    workers_.emplace_back(&AsyncFileWriter::Worker, this);
  }
}

void AsyncFileWriter::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  producer_cv_.notify_all();
  for (auto &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
  running_ = false;
}

void AsyncFileWriter::Enqueue(AsyncWriteTask task) {
  bool need_start = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    producer_cv_.wait(
        lock, [&] { return stop_ || tasks_.size() < max_queue_depth_; });
    if (stop_) {
      return;
    }
    tasks_.push(std::move(task));
    need_start = !running_;
  }
  if (need_start) {
    Start();
  } else {
    cv_.notify_one();
  }
}

void AsyncFileWriter::Worker() {
  SetCurrentThreadName("if-kv-writer");
  while (true) {
    AsyncWriteTask task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
      producer_cv_.notify_one();
    }
    std::ofstream out(task.path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      continue;
    }
    out.write(task.buffer.data(),
              static_cast<std::streamsize>(task.buffer.size()));
  }
}

} // namespace inferflux
