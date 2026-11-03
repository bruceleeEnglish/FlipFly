/**
	@author:	Yuxiang Zeng
	@email: 	yxzeng@buaa.edu.cn
	@date:		2024.09.28
*/
#ifndef UTILS_FILE_IO_H  
#define UTILS_FILE_IO_H  
  
#include <fstream>
#include <string>
#include <memory>
#include <iostream>
#include <cstdlib>

#include "DataType.hpp"
  

// READ & DUMP vector data from specific file
void ReadVectorData(const std::string& file_name, std::vector<VectorDataType>& data_list);
void DumpVectorData(const std::string& file_name, const std::vector<VectorDataType>& data_list);

void ReadVectorData(const std::string& file_name, float ** data, size_t &n, size_t &d);

// READ & DUMP vector queries from specific file
void ReadVectorQuery(const std::string& file_name, std::vector<VectorDataType>& query_data_list, std::vector<size_t>& query_k_list);
void DumpVectorQuery(const std::string& file_name, const std::vector<VectorDataType>& query_data_list, const std::vector<size_t>& query_k_list);

// READ & DUMP IP addresses of data silos from specific file
void ReadSiloIPaddr(const std::string& file_name, std::vector<std::string>& silo_ipaddr_list);
void DumpSiloIPaddr(const std::string& file_name, const std::vector<std::string>& silo_ipaddr_list);

// READ & DUMP ground truth from specific file
void ReadGroundTruth(const std::string& file_name, std::vector<std::vector<VidType>>& answer_list);
void DumpGroundTruth(const std::string& file_name, const std::vector<std::vector<VidType>>& answer_list);

// Split vector data into several files
void SplitVectorData(const std::string& file_name, const int silo_num, const std::string& des_file_path="", const std::vector<VectorDataType> data_list={});
void SplitVectorDataForDigData(const std::string& file_name, const int silo_num, const std::string& des_file_path="");
// Evaluate the query answer based on the ground truth
void EvaluateAnswer(const std::string& answer_filename, const std::string& truth_filename);
void EvaluateAnswer(std::vector <std::vector<VidType>>& answer_list, std::vector <std::vector<VidType>>& truth_list, int K=256);

#endif // UTILS_FILE_IO_H