#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>
using namespace std;


struct CompareKeys {
    bool operator()(const int a, const int b) const {
        return a > b; 
    }
};


//Function for decompressing the dictionary
map<int,set<int>,CompareKeys>  get_decompressed_dict(std::map<int, std::set<int>,CompareKeys> original_dict){
    //sort the keys of the original dictionary in descending order
    map<int,set<int>,CompareKeys> decompressed_dict;
    for (const auto& pair:original_dict){
        int present_label=pair.first;
        for (auto item:pair.second){
            if (item>=0){
                decompressed_dict[present_label].insert(item);
            }else{
                for (auto i:decompressed_dict[item]){
                    decompressed_dict[present_label].insert(i);
                }
            }
        }
    }
    return decompressed_dict;
}

void d_helper(std::vector<std::vector<int>>& transactions, std::string path_to_decompressed_datafile){
    // Opening a output data file
    std::ofstream outfile;
    outfile.open(path_to_decompressed_datafile, std::ofstream::out | std::ofstream::trunc);

    for(auto transaction : transactions){
        for(auto item : transaction){
            outfile << item << ' ';
        }
        outfile << '\n';
    }
    outfile.close();
}
void decompress(std::string decompressed_file_path, std::string compressed_file_path)
{
    std::string line; int num ;
    std::map<int, std::set<int>,CompareKeys> conversion_dictionary;
    map<int,set<int>,CompareKeys> decompressed_dict;
    std::ifstream compressed_file(compressed_file_path);

    if (!compressed_file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }

   vector<vector<int>>transactions;

    bool flag = true;
    while(std::getline(compressed_file, line))
    {
        if(flag && line.length()>0)
        {
            // store dictionary
            //cout<<"IN dict"<<endl;
            int key;
            std::istringstream iss(line);
            iss >> key ;;
            while(iss >> num)
            {
                conversion_dictionary[key].insert(num);
            }
        }
        else if (line.length()==0){
            //cout<<"Reached here"<<endl;
            flag = false;
            // decompress dictionary
            decompressed_dict=get_decompressed_dict(conversion_dictionary);
        }
        else{
            // process compressed transactions
            std::istringstream iss(line);
            vector<int>t;
            while(iss>>num)
            {
                if(num >= 0)
                {
                    t.push_back(num);
                }else{
                    for(auto ele : decompressed_dict[num])
                        t.push_back(ele);
                }
            }
            transactions.push_back(t);
        }
      
    }

    compressed_file.close();
    d_helper(transactions,decompressed_file_path);
}

int main(int argc, const char *argv[])
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string compressed_file_path = argv[1];
    std::string decompressed_file_path = argv[2];
    
    decompress(decompressed_file_path, compressed_file_path);
    
    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the duration in seconds
    std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;

    return EXIT_SUCCESS;
}