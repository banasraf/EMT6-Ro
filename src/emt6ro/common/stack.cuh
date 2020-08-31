#ifndef EMT6RO_COMMON_STACK_CUH_
#define EMT6RO_COMMON_STACK_CUH_
#include <cstdint>
#include "cuda-utils.h"

namespace emt6ro {
  
template <typename Element, typename Index = uint32_t>
class StackView {
 public:
  static_assert(sizeof(Element) == sizeof(Index), 
                "Element type and Index type should have an equal size.");

  __host__ __device__
  StackView(void *mem)
    : size_(reinterpret_cast<Index*>(mem))
    , data_(reinterpret_cast<Element*>(size_ + 1)) {}

  __host__ __device__
  Element &operator[](Index i) {
    return data_[i];
  }
  
  __host__ __device__
  const  Element &operator[](Index i) const {
    return data_[i];
  }
  
  __host__ __device__
  const Index &size() const {
    return *size_;
  }
  
  __host__ __device__
  Index &size() {
    return *size_;
  }
  
  __host__ __device__
  void push(Element elem) {
    data_[(*size_)++] = elem;
  }
  
 private:
  Index *size_;
  Element *data_;
};

template <typename Element, typename Index = uint32_t>
class StackDevIterable {
 public:
  class Iterator {
    Element *ptr_;

   public:
    Iterator() = default;
    
    __device__
    Iterator(Element *ptr): ptr_(ptr) {}
    
    __device__
    Element &operator*() {
      return *ptr_;
    }
    
    __device__
    bool operator==(Iterator rhs) {
      return ptr_ == rhs.ptr_;
    }
    __device__
    bool operator!=(Iterator rhs) {
      return ptr_ != rhs.ptr_;
    }
    
    __device__
    Iterator &operator++() {
      ptr_ += blockVol();
      return *this;
    }
    
    __device__
    Iterator operator++(int) {
      Iterator copy = *this;
      ++(*this);
      return copy;
    }
  };

  __device__
  Iterator begin() {
    return Iterator(&stack_[threadId()]);
  }

  __device__
  Iterator end() {
    ptrdiff_t size = stack_.size() - threadId();
    if (size <= 0) {
      size = 0;
    } else {
      size = div_ceil(size, blockVol()) * blockVol();
    }
    return Iterator(&stack_[threadId() + size]);
  }
  
  __device__
  StackDevIterable(StackView<Element, Index> &stack): stack_(stack) {}

 private:
  StackView<Element, Index> stack_;
};

template <typename Element, typename Index = uint32_t>
__device__
StackDevIterable<Element, Index> dev_iter(StackView<Element, Index> &stack) {
  return StackDevIterable<Element, Index>(stack);
}

} // namespace emt6ro
#endif  // EMT6RO_COMMON_STACK_CUH_
