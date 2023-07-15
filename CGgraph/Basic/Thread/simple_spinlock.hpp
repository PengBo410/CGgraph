#ifndef SIMPLE_SPINLOCK_HPP
#define SIMPLE_SPINLOCK_HPP

#include <assert.h>
#include <mutex>

class simple_spinlock {
private:
	// mutable not actually needed
	mutable volatile char spinner;
public:
	/// constructs a spinlock
	simple_spinlock() {
		spinner = 0;
	}

	/** Copy constructor which does not copy. Do not use!
	Required for compatibility with some STL implementations (LLVM).
	which use the copy constructor for vector resize,
	rather than the standard constructor.    */
	simple_spinlock(const simple_spinlock&) {
		spinner = 0;
	}

	// not copyable
	void operator=(const simple_spinlock& m) { }


	/// Acquires a lock on the spinlock
	inline void lock() const {
		while (spinner == 1 || __sync_lock_test_and_set(&spinner, 1));
	}
	/// Releases a lock on the spinlock
	inline void unlock() const {
		__sync_synchronize();
		spinner = 0;
	}
	/// Non-blocking attempt to acquire a lock on the spinlock
	inline bool try_lock() const {
		return (__sync_lock_test_and_set(&spinner, 1) == 0);
	}
	~simple_spinlock() {
		assert(spinner == 0);
	}
};



class mutex {
public:
	// mutable not actually needed
	mutable pthread_mutex_t m_mut;
	/// constructs a mutex
	mutex() {
		int error = pthread_mutex_init(&m_mut, NULL);
		assert(!error);
	}
	/** Copy constructor which does not copy. Do not use!
		Required for compatibility with some STL implementations (LLVM).
		which use the copy constructor for vector resize,
		rather than the standard constructor.    */
	mutex(const mutex&) {
		int error = pthread_mutex_init(&m_mut, NULL);
		assert(!error);
	}

	~mutex() {
		int error = pthread_mutex_destroy(&m_mut);
		assert(!error);
	}

	// not copyable
	void operator=(const mutex& m) { }

	//【非阻塞锁】
	//这个操作是阻塞调用的，也就是说，如果这个锁此时正在被其它线程占用， 那么 pthread_mutex_lock() 调用会进入到这个锁的排队队列中，并会进入阻塞状态， 直到拿到锁之后才会返回。
	inline void lock() const {
		int error = pthread_mutex_lock(&m_mut);
		// if (error) std::cout << "mutex.lock() error: " << error << std::endl;
		assert(!error);
	}
	//【阻塞锁】
	//如果不想阻塞，而是想尝试获取一下，如果锁被占用咱就不用，如果没被占用那就用， 这该怎么实现呢？可以使用 pthread_mutex_trylock() 函数。 
	//这个函数和 pthread_mutex_lock() 用法一样，只不过当请求的锁正在被占用的时候， 不会进入阻塞状态，而是立刻返回，并返回一个错误代码 EBUSY，意思是说， 有其它线程正在使用这个锁。
	inline void unlock() const {
		int error = pthread_mutex_unlock(&m_mut);
		assert(!error);
	}
	/// Non-blocking attempt to acquire a lock on the mutex
	inline bool try_lock() const {
		return pthread_mutex_trylock(&m_mut) == 0;
	}
	//friend class conditional;
}; // End of Mutex


#endif
