#ifndef CPJ_TIME_HPP
#define CPJ_TIME_HPP

#ifdef __GNUC__
#include <sys/time.h>
#endif
#ifdef _MSC_VER 
#include <time.h>
#endif
#include <stdio.h>
#include <iostream>
#include <signal.h>

#define	SIGALRM		14
#define ITIMER_REAL_ 0

#ifdef _MSC_VER
	/*
* Structure used in select() call, taken from the BSD file sys/time.h.
*/
struct timeval {
	long tv_sec;           /* seconds */
	long tv_usec;          /* and microseconds */
};

int gettimeofday(struct timeval* tp, void* tzp);

#endif /* _MSC_VER */

//void alarm_wakeup(int i);
//class hundredms_timer {
//public:
//	hundredms_timer() {
//		stop = false;
//		tout_val.it_interval.tv_sec = 0;
//		tout_val.it_interval.tv_usec = 0;
//		tout_val.it_value.tv_sec = 0;
//		tout_val.it_value.tv_usec = 50000;
//		signal(SIGALRM, alarm_wakeup); /* set the Alarm signal capture */
//		setitimer(ITIMER_REAL_, &tout_val, 0);
//
//		ti.start();
//	}
//	size_t ctr;
//	timer ti;
//	struct itimerval tout_val;
//	bool stop;
//
//	~hundredms_timer() {
//		stop = true;
//		signal(SIGALRM, SIG_IGN);
//	}
//};
//
//void alarm_wakeup(int i) {
//	if (hmstimer.stop) return;
//	signal(SIGALRM, alarm_wakeup);
//	// compute the desired time till the next 100ms tick by using a real timer call
//	double realtime = hmstimer.ti.current_time();
//	// round down
//	hmstimer.ctr = (size_t)(realtime * 10);
//	setitimer(ITIMER_REAL_, &(hmstimer.tout_val), 0);
//}

class timer {
private:
	timeval start_time_;

	//hundredms_timer hmstimer;
public:
	
	inline timer() { start(); }


	inline void start() { gettimeofday(&start_time_, NULL); }


	inline double current_time() const {
		timeval current_time;
		gettimeofday(&current_time, NULL);
		double answer =
			// (current_time.tv_sec + ((double)current_time.tv_usec)/1.0E6) -
			// (start_time_.tv_sec + ((double)start_time_.tv_usec)/1.0E6);
			(double)(current_time.tv_sec - start_time_.tv_sec) +
			((double)(current_time.tv_usec - start_time_.tv_usec)) / 1.0E6;
		return answer;
	} // end of current_time

	inline timeval curTimeval()
	{
		timeval current_time;
		gettimeofday(&current_time, NULL);
		return current_time;
	}


	inline double current_time_millis() const { return current_time() * 1000; }
	inline double get_time_ms() const { return current_time() * 1000; }
	inline double get_time() const { return current_time(); }

	/**
	 * \brief Get the number of seconds (as a floating point value)
	 * since the Unix Epoch.
	 */
	static double sec_of_day() {
		timeval current_time;
		gettimeofday(&current_time, NULL);
		double answer =
			(double)current_time.tv_sec + ((double)current_time.tv_usec) / 1.0E6;
		return answer;
	} // end of sec_of_day


	static size_t usec_of_day() {
		timeval current_time;
		gettimeofday(&current_time, NULL);
		size_t answer =
			(size_t)current_time.tv_sec * 1000000 + (size_t)current_time.tv_usec;
		return answer;
	} // end of usec_of_day


	 /**
	  * Precision of deciseconds
	  */
	float approx_time_seconds() {
		return 0;
		// float(hmstimer.ctr) / 10;
	}

	/**
	 * Precision of deciseconds
	 */
	size_t approx_time_millis() {
		return 0;
		//return hmstimer.ctr * 100;
	}


#ifdef __GNUC__
	static void sleep(size_t sleeplen)
	{
		struct timespec timeout;
		timeout.tv_sec = sleeplen;
		timeout.tv_nsec = 0;
		while (nanosleep(&timeout, &timeout) == -1);
	}
#endif
	


#ifdef __GNUC__
	static void sleep_ms(size_t sleeplen)
	{
		struct timespec timeout;
		timeout.tv_sec = sleeplen / 1000;
		timeout.tv_nsec = (sleeplen % 1000) * 1000000;
		while (nanosleep(&timeout, &timeout) == -1);
	}
#endif

	void printFormatTime()
	{
		time_t tt;
		time(&tt);
		tt = tt + 8 * 3600;  // transform the time zone
		tm* t = gmtime(&tt);

		printf("%d-%02d-%02d %02d:%02d:%02d\n",
			t->tm_year + 1900,
			t->tm_mon + 1,
			t->tm_mday,
			t->tm_hour,
			t->tm_min,
			t->tm_sec);
	}

	std::string getFormatTime()
	{
		time_t tt;
		time(&tt);
		tt = tt + 8 * 3600;
		tm* t = gmtime(&tt);

		char str[1024];
		sprintf(str, "%d-%02d-%02d %02d:%02d:%02d\n",
			t->tm_year + 1900,
			t->tm_mon + 1,
			t->tm_mday,
			t->tm_hour,
			t->tm_min,
			t->tm_sec);

		return std::string(str);
	}


}; // end of Timer


unsigned long long estimate_ticks_per_second();

#if defined(__i386__)                           
static inline unsigned long long rdtsc(void)
{
	unsigned long long int x;
	__asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
	return x;
}
#elif defined(__x86_64__)                     
static inline unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
	return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#else
static inline unsigned long long rdtsc(void) {
	return 0;
}
#endif




#endif // !CPJ_TIME_HPP
