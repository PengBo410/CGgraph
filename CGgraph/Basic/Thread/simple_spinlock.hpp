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

	
	inline void lock() const {
		int error = pthread_mutex_lock(&m_mut);
		// if (error) std::cout << "mutex.lock() error: " << error << std::endl;
		assert(!error);
	}

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
