#ifndef __LOGGER_H__
#define __LOGGER_H__

// if you want logging you need to define WITH_LOGGING, 
// then call Logger::log("any text"); once in your main thread 
// before using the classes in this file. It assures 
// Logger::instance() to run properly (static boost::mutex s_mu is created un-interruptedly).
#ifdef WITH_LOGGING 
#define _WITH_LOGGING
#endif

// debug classes...
struct Logger 
{
#ifdef _WITH_LOGGING
// call this once at start of application (main thread) to make sure s_mu is proper
inline static boost::mutex & instance()
{
	static boost::mutex s_mu;
	return(s_mu);
}
#endif

inline static void log(const char * buf)
{
#ifdef _WITH_LOGGING
	// can not protect if an outside thread that does not know the mutex and accesses std::cout directly
	boost::mutex::scoped_lock lock(instance());
	std::cout << buf << std::endl << std::flush;
#endif
}
};


#endif
