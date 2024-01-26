#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>
#include <map>
#include <algorithm>
using namespace std;
#include "fptree.h"

void helper(vector<Pattern> patterns_mined,vector<Transaction>&transactions,map<set<Item>,long long>&compression_dict,long long& label){
    auto patternComparator = [](const Pattern& a, const Pattern& b) {
        return a.first.size() > b.first.size(); 
    };
    sort(patterns_mined.begin(),patterns_mined.end(),patternComparator);
    for (int i=0;i<transactions.size();i++){
        set<Item> temp_transaction; 
        for (int k=0;k<transactions[i].size();k++){
            temp_transaction.insert(transactions[i][k]);
        }
        for (int j=0;j<patterns_mined.size();j++){
            if (patterns_mined[j].second>1 && patterns_mined[j].first.size()>1){
                if (includes(temp_transaction.begin(),temp_transaction.end(),patterns_mined[j].first.begin(),patterns_mined[j].first.end())){
                if (compression_dict.count(patterns_mined[j].first)==0){
                    compression_dict[patterns_mined[j].first]=label;
                    label--;
                }
                for (auto ele:patterns_mined[j].first){
                    temp_transaction.erase(ele);
                }
                temp_transaction.insert(compression_dict[patterns_mined[j].first]);
            }
            }
            
        }
        transactions[i]=vector<Item> (temp_transaction.begin(),temp_transaction.end());
    }
}
void compress_data(string file_path,string compressed_file_path){
    std::string line ; int num ;
    std::ifstream input_file(file_path);
    std::vector<Transaction> transactions; 
    int total_initial_terms=0;
    int total_transactions=0;
    long long  label=-1;
    int total_unique=0;
    map<set<Item>,long long> compression_dict;
    if (!input_file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }
   
     //  Divide into buckets based on transaction size
    std::map<int, std::vector<Transaction>> size_buckets;
    
    while (std::getline(input_file, line)) {
        Transaction transaction;
        std::istringstream iss(line);
        map<int,bool>mp;
        while(iss >> num)
        {
            Item item{num};
            if (mp.count(item)==0){
                total_unique++;
            }
            transaction.push_back(item);
            total_initial_terms ++ ;
        }
        transactions.push_back(transaction);

        //Divide transactions into buckets based on size- our goal is obtain almost equal distribution in all buckets
        int transaction_size = transaction.size();
        if (transaction_size <= 100 &&(size_buckets.count(1)==0 || (size_buckets.count(1)>0 && size_buckets[1].size()<100000)) ) {
            size_buckets[1].push_back(transaction);
          
        } else if (transaction_size <= 1000 && (size_buckets.count(2)==0 || (size_buckets.count(2)>0 && size_buckets[2].size()<100000))) {
            size_buckets[2].push_back(transaction);
        } else if (transaction_size <= 10000 && (size_buckets.count(3)==0 || (size_buckets[3].size()<100000))) {
            size_buckets[3].push_back(transaction);
        } else if (transaction_size <= 100000 &&  (size_buckets.count(4)==0 || (size_buckets[4].size()<100000))) {
            size_buckets[4].push_back(transaction);
        }else{
            size_buckets[5].push_back(transaction);
        }
    }
    input_file.close();
    




    
    //mining over each bucket
   for ( auto [size_bucket, bucket_transactions] : size_buckets){
    std::cout<<"Size of bucket "<<bucket_transactions.size()<<endl;
  }

     auto algo_start_time = std::chrono::high_resolution_clock::now();

     int limit = 240;
     for ( auto [size_bucket, bucket_transactions] : size_buckets){
        std::cout<<"Size of bucket "<<bucket_transactions.size()<<endl;

        limit += 60;
        float start_threshold=1.0;
        float step_one=0.1;
        float step_two=0.05;
        bool flag=true;

        auto start_time = std::chrono::high_resolution_clock::now();
        while   (start_threshold>0)    {

            auto temp_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(temp_time - start_time);
            
            if (elapsed_time > std::chrono::seconds(limit))
            {
                std::cout << "Algorithm exceeded the maximum allowed execution time." << std::endl;
                break;
            }

            std::cout<<"Processing support : "<<start_threshold<<endl;
            const FPTree t1(bucket_transactions,start_threshold);
            auto mining_time = std::chrono::high_resolution_clock::now();
            auto patterns_mined=mine_fptree(t1, &mining_time);

            auto current_time = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - mining_time);

            std::cout<<"No. of patterns mined "<<patterns_mined.size()<<endl;

            helper(patterns_mined,bucket_transactions,compression_dict,label);

            size_buckets[size_bucket]=bucket_transactions;

            if (elapsed_time > std::chrono::seconds(120))
            {
                std::cout << "Function exceeded the maximum allowed execution time." << std::endl;
                break;
            }
            
            if (start_threshold>0.2){
                if (flag){
                    start_threshold-=step_one;
                    flag=false;
                }else{
                    flag=true;
                    start_threshold-=step_two;
                }
            }else if (start_threshold>0.0125){
               
                start_threshold-=0.0125;
            }else{
                start_threshold-=0.0005;
            }

        } 
        
    }
    
    
    int final_items=0;
    std::ofstream outFile; 
    outFile.open(compressed_file_path,std::ofstream::out | std::ofstream::trunc);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return ;
    }
    //Printing the dictionary first
    for(auto pair : compression_dict)
    {
        outFile << pair.second << " "; // label 
        final_items ++ ;
        for(auto ele : pair.first)
        {
            outFile << ele << " " ;
            final_items ++ ;
        }
        outFile << "\n";
    }
    outFile << "\n" ;
    for ( auto [size_bucket, bucket_transactions] : size_buckets){
        for (auto transaction:bucket_transactions){
            for (auto item:transaction){
                outFile << item << " ";
                final_items ++ ;
            }
            outFile << "\n";
        }
       
    }
    

    outFile.close();

    // Print Statistics
    std::cout << "Initial Items in dataset : " << total_initial_terms  << "\n";
    std::cout << "Final Items in compressed data : " << final_items << "\n";
    std::cout << "Amount of Compression achieved : " << (100.0 - float(final_items)/total_initial_terms*100)  << "\n";

}

int main(int argc, const char *argv[])
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string file_path = argv[1];
    std::string compressed_file_path = argv[2];
    
    compress_data(file_path, compressed_file_path);
    
    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the duration in seconds
    std::cout << "Time taken: " << elapsedTime.count() / 1000.0 << " seconds" << std::endl;

    return EXIT_SUCCESS;
}
