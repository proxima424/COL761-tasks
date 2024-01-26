#ifndef FPTREE_HPP
#define FPTREE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>


using Item = int;
using Transaction = std::vector<Item>;
using TransformedPrefixPath = std::pair<std::vector<Item>, uint64_t>;
using Pattern = std::pair<std::set<Item>, int>;


struct FPNode {
    const Item item;
    int frequency;
    std::shared_ptr<FPNode> next;
    std::weak_ptr<FPNode> parent;
    std::map<Item, std::shared_ptr<FPNode>> children;

    FPNode(const Item&, const std::shared_ptr<FPNode>&);
};

struct FPTree {
    public:
        std::shared_ptr<FPNode> root;
        float minimum_support_threshold;
        std::map<Item, int> item_frequencies;
        std::map<Item, std::shared_ptr<FPNode>> header_table;
        std::map<Item, std::shared_ptr<FPNode>> last_node_in_header_table;
        FPTree(const std::vector<Transaction>&, float);
        int total_transactions;
        int total_items;
        FPTree (const std::vector<TransformedPrefixPath>&, float );
    
        bool empty() const;
};


std::vector<Pattern> mine_fptree(const FPTree& fptree, std::chrono::high_resolution_clock::time_point* start_time);
std::vector<std::pair<std::vector<Item>, uint64_t>> get_hashmap(const std::vector<Transaction>&, float);

#endif  // FPTREE_HPP