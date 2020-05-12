#ifndef CARE_KVTABLE_HPP
#define CARE_KVTABLE_HPP

#include <vector>
#include <utility>

namespace care{

    template<class Key, class Value>
    class KVTable{
    public:
        KVTable() = default;
        KVTable(std::size_t capacity){
            keys.reserve(capacity);
            values.reserve(capacity);
        }

        KVTable(const KVTable&) = default;
        KVTable(KVTable&&) = default;
        KVTable& operator=(const KVTable&) = default;
        KVTable& operator=(KVTable&&) = default;

        void addPair(Key key, Value value){
            keys.emplace_back(std::move(key));
            values.emplace_back(std::move(value));
        }

        std::vector<Key>& getKeys() noexcept{
            return keys;
        }

        std::vector<Value>& getValues() noexcept{
            return values;
        }

        const std::vector<Key>& getKeys() const noexcept{
            return keys;
        }

        const std::vector<Value>& getValues() const noexcept{
            return values;
        }

    private:
        std::vector<Key> keys;
        std::vector<Values> values;        
    };

} //namespace care

#endif
