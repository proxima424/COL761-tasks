#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>
#include "fptree.h"
#include <iostream>

using namespace std;
FPNode::FPNode(const Item& item, const std::shared_ptr<FPNode>& parent) :
    item( item ), frequency( 1 ), next( nullptr ), parent( parent ), children()
{
}


FPTree::FPTree(const std::vector<Transaction>& transactions, float support) :
    root( std::make_shared<FPNode>( Item{}, nullptr ) ), header_table(),item_frequencies()
    //minimum_support_threshold( minimum_support_threshold )
{
    // scan the transactions counting the frequency of each item
    this->total_items=0;
    this->total_transactions = transactions.size();
    for ( const Transaction& transaction : transactions ) {
        for ( const Item& item : transaction ) {
            ++item_frequencies[item];
            this->total_items+=1;
        }
    }
     this->minimum_support_threshold = int(support*(this->total_transactions));

    // keep only items which have a frequency greater or equal than the minimum support threshold
    for ( auto it = item_frequencies.cbegin(); it != item_frequencies.cend(); ) {
        const int item_frequency = (*it).second;
        if ( item_frequency < minimum_support_threshold ) { item_frequencies.erase( it++ ); }
        else { ++it; }
    }

    //using intution of SRMine
    map<Transaction,int> hash_map;
    for (const Transaction& transaction : transactions ){
        Transaction pruned_transcation;
        for(auto ele : transaction)
        {
            if(item_frequencies.count(ele))
            {
                pruned_transcation.push_back(ele);
            }
        }
        sort(pruned_transcation.begin(), pruned_transcation.end(), [this](Item a, Item b)
        {
            return this->item_frequencies[a] > this->item_frequencies[b];
        });
        hash_map[pruned_transcation]++;
    }
    

    // starting tree construction
    for ( const auto& pair : hash_map ) {
        Transaction transaction=pair.first;
        int freq=pair.second;
        auto curr_node=root;
        for (const Item& item:transaction){
            //check if item is present in one of the children of current node

            //if not present
            if (curr_node->children.count(item)==0){
                const auto new_fp_node_child = make_shared<FPNode>( item, curr_node );
                new_fp_node_child->frequency=freq;
                // add the new node to the tree
                curr_node->children[item] = (new_fp_node_child) ;

                // update the node-link structure
                if ( header_table.count(item)) {
                    last_node_in_header_table[item]->next = new_fp_node_child;
                    last_node_in_header_table[item] = new_fp_node_child;
                }
                else {
                    header_table[item] = new_fp_node_child;
                    last_node_in_header_table[item] = new_fp_node_child;
                }

                // advance to the next node of the current transaction
                curr_node = new_fp_node_child;
            }else{
                // the child exist, increment its frequency
                auto fp_node_child = curr_node->children[item];
                fp_node_child->frequency=fp_node_child->frequency+freq;

                // advance to the next node of the current transaction
                curr_node = fp_node_child;
            }
        }
    }
}
bool FPTree::empty() const
{
    return root->children.empty();
}

// these functions test if a tree is a single path A --> B --> C --> D --> E -- > -|
bool containts_single_path(const std::shared_ptr<FPNode>& fpnode)
{
    if (fpnode -> children.size() == 0)
    {
        return true;
    }
    if(fpnode -> children.size() > 1)
    {
        return false;
    }
    return containts_single_path((*(fpnode->children.begin())).second);
}

bool containts_single_path(const FPTree& fptree)
{
    return fptree.empty() || containts_single_path(fptree.root);
}

std::vector<Pattern> mine_fptree(const FPTree& fptree, std::chrono::high_resolution_clock::time_point* start_time )
{
    
    
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - *start_time);
    if (elapsed_time > std::chrono::seconds(120))
    {
        // std::cout << "Function exceeded the maximum allowed execution time." << std::endl;
        return {};
    }
    
    if (fptree.empty()){ return {};}

    else if (containts_single_path(fptree))
    {
        // generate all possible combinationso of items in the trees

        std::vector<Pattern> single_path_patterns;

        // for each node in tree
        auto fpnode = (*(fptree.root->children.begin())).second;
        while (fpnode)
        {
            const Item& item = fpnode -> item ;
            const uint64_t frequency = fpnode -> frequency ;

            // add a pattern formed by only the current node
            // this will be frequent because we removed infrequent single items from the 
            Pattern single_item_pattern{{item}, frequency};

            // create a new pattern by adding the item of the current node to all the previously generated patterns
            int curr_size = single_path_patterns.size();
            for( int idx = 0 ; idx < curr_size ; idx ++  )
            {
                Pattern new_pattern = single_path_patterns[idx];
                new_pattern.first.insert(item);
                new_pattern.second = frequency;
                single_path_patterns.push_back(new_pattern);
            }
            single_path_patterns.push_back(single_item_pattern);
            if (fpnode->children.size())
            {
                fpnode = (*(fpnode->children.begin())).second;
            }else{
                fpnode = nullptr ;
            }
        }
       // cout<<single_path_patterns.size()<<endl;
        return single_path_patterns;
    }
    else{
        // generaate conditional fptrees for each different item in the fptree
        std::vector<Pattern> multi_path_patterns ;

        for (const auto & pair : fptree.header_table)
        {
            const Item& curr_item = pair.first;

            std::vector<TransformedPrefixPath> conditional_pattern_base;

            // for each node in the header table corresponding to the current item
            auto item_node = pair.second;
            while(item_node) // this loop iterates over all nodes of an item from the header table 
            {
                const uint64_t path_starting_fpnode_frequency = item_node->frequency;

                auto curr_path_fpnode = item_node->parent.lock(); // starting node of path
                // check if curr_path_fpnode is already the root of the fptree
                if(curr_path_fpnode->parent.lock())
                {
                    TransformedPrefixPath transformed_prefix_path{{}, path_starting_fpnode_frequency};
                    while(curr_path_fpnode->parent.lock())
                    {
                        transformed_prefix_path.first.push_back(curr_path_fpnode->item);
                        curr_path_fpnode = curr_path_fpnode->parent.lock(); // move up the path
                    }
                    conditional_pattern_base.push_back(transformed_prefix_path);
                }

                // advance to the next node in the header table
                item_node = item_node ->next;
            }


            std::vector<Transaction> temp_trans;
            for ( const TransformedPrefixPath& tfp : conditional_pattern_base ) {
                for (int k=0;k<tfp.second;k++){
                    Transaction t;
                    for ( const Item& item : tfp.first ){
                        t.push_back(item);
                    }
                    temp_trans.push_back(t);
                }
                
            }

            const FPTree conditional_fptree(temp_trans, fptree.minimum_support_threshold/temp_trans.size());
            //const FPTree conditional_fptree(conditional_pattern_base,fptree.minimum_support_threshold);
            // this is a recursive function call
            // gets the frequent patters in the conditional FPTree 
            std::vector<Pattern> conditional_patterns = mine_fptree(conditional_fptree, start_time); // recursive function
        
            // construct patterns relative to the current item using both the current item and the conditional patterns
            std::vector<Pattern> curr_item_patterns;

            // the first pattern is made only by the current item
            // compute the frequency of this pattern by summing the frequency of the nodes which have the same item (follow the node links)
            uint64_t curr_item_frequency = (*fptree.item_frequencies.find(curr_item)).second;
            // add the pattern as a result
            Pattern pattern{{ curr_item }, curr_item_frequency }; // this is a frequent pattern since we only considered frequent items 
            curr_item_patterns.push_back( pattern );

            // other patterns are generated recursively by adding the current item to the recursively generated patterns
            for (const Pattern& pattern : conditional_patterns)
            {
                Pattern new_pattern{pattern};
                new_pattern.first.insert(curr_item);
                new_pattern.second = pattern.second;
                curr_item_patterns.push_back({new_pattern});
            }

            // join the patterns generated by the current item with all the other items of the fptree
            multi_path_patterns.insert(multi_path_patterns.end(), curr_item_patterns.cbegin(), curr_item_patterns.cend() );

            if (multi_path_patterns.size() > 50000) {
                return multi_path_patterns;
            }
           
        }

        return multi_path_patterns;
    }
}