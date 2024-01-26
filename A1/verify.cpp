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

using namespace std;

set<set<int>>read_trans(string file_path){
    std::string line ; int num ;
    set<set<int>>transactions;
    std::ifstream input_file(file_path);
    
    while (std::getline(input_file, line)) {
        set<int> transaction;
        std::istringstream iss(line);
        while(iss >> num)
        {
            transaction.insert(num);
        }
        transactions.insert(transaction);
    }
    return transactions;
}

void compute_error(string file1,string file2){
    set<set<int>>transactions1=read_trans(file1);
    set<set<int>>transactions2=read_trans(file2);
    if (transactions1.size()!=transactions2.size()){
        std::cout<<"Lossy compression"<<endl;
    }else{

        // Iterate through the sets in set1
    for (const auto& subSet1 : transactions1) {
        // Check if the current set in set1 exists in set2
        if (transactions2.find(subSet1) == transactions2.end()) {
            cout<<"Error in compression"<<endl;
            for (auto it:subSet1){
                cout<<it<<" ";
            }
            return;
        }
    }
    for (const auto& subSet2 : transactions2) {
        // Check if the current set in set1 exists in set2
        if (transactions1.find(subSet2) == transactions1.end()) {
            cout<<"Error in compression2"<<endl;
            return;
        }
    }
    }
    cout<<"No Error"<<endl;
}




int main(int argc, const char *argv[]){
    string file1 = argv[1];
    string file2 = argv[2];
    compute_error(file1,file2);
    return 0;
}


