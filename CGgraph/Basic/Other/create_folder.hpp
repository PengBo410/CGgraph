#pragma once

#include "../Platform/platform_def.hpp"
#include "../Console/console.hpp"

#if defined (PLATFORM_WIN)

#include <io.h>     //access
#include <assert.h>
#include <iostream>
#include <direct.h> //_mkdir

	//注意：path必须以"/"结尾，如：const char* path = "C:\\Users\\yyj\\Desktop\\vs_log\\"
	bool createFloder(const char* path) {

		// >= 0 说明目录存在，小于0不存在
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

    //注意：path必须以"/"结尾，如：const char* path = "logger/"
	bool createFloder(const char* path)
	{
		// >= 0 说明目录存在，小于0不存在。 
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
        //如果已经存在则不在做出任何操作
		else
		{
			//printf("[%s] floder is already existed \n", path);
			return true;
		}	
	}
#endif