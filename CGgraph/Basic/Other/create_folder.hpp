#pragma once

#include "../Platform/platform_def.hpp"
#include "../Console/console.hpp"

#if defined (PLATFORM_WIN)

#include <io.h>     //access
#include <assert.h>
#include <iostream>
#include <direct.h> //_mkdir


	bool createFloder(const char* path) {

		if (_access(path, 0) < 0) {

			int isCreate = _mkdir(path);
			if (!isCreate)
			{
				Msg_info("Create Folder:[%s]", path);
				return true;
			}
			else
			{
				assert_msg(false, "create Folder failed! The error code : %d \n", isCreate);
				return false;
			}
		}
		else
		{
			//printf("[%s] floder is already existed \n", path);
			return true;
		}
	}



#elif defined (PLATFORM_LINUX)

#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>
#include <unistd.h> //access
#include <iostream>

    
	bool createFloder(const char* path)
	{
		 
		if (access(path, F_OK) < 0)
		{
			int isCreate = mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
			if (!isCreate)
			{
				
                Msg_info("Create Folder:[%s]", path);
				return true;
			}				
			else 
			{
                assert_msg(false, "create Folder failed! The error code : %d \n", isCreate);
				return false;
			}				
		}
        
		else
		{
			//printf("[%s] floder is already existed \n", path);
			return true;
		}	
	}
#endif