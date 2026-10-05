// include/strata/core/session_registry.hpp - Multi-Session Persistent Cold-Start Registry
//
// Cross-Engine Attribution:
//   - SGLang: Persistent session checkpointing and zero-prefill session resumption
//
#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

struct SessionRecord {
    std::string session_id;
    int64_t root_node_id = 0;
    int64_t total_tokens = 0;
    uint64_t last_accessed_sec = 0;
    std::vector<int32_t> prefix_tokens;
};

class SessionRegistry {
public:
    explicit SessionRegistry(const std::string& registry_file = "")
        : registry_file_(registry_file) {
        if (!registry_file_.empty()) {
            load_from_disk();
        }
    }

    ~SessionRegistry() {
        if (!registry_file_.empty()) {
            save_to_disk();
        }
    }

    void register_session(const std::string& session_id,
                          int64_t root_node_id,
                          int64_t total_tokens,
                          const std::vector<int32_t>& prefix_tokens = {}) {
        if (session_id.empty()) return;
        std::unique_lock<std::shared_mutex> lock(mutex_);
        SessionRecord rec;
        rec.session_id = session_id;
        rec.root_node_id = root_node_id;
        rec.total_tokens = total_tokens;
        rec.prefix_tokens = prefix_tokens;
        rec.last_accessed_sec = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        sessions_[session_id] = std::move(rec);
    }

    bool lookup_session(const std::string& session_id,
                        int64_t& out_root_node_id,
                        int64_t& out_total_tokens,
                        std::vector<int32_t>* out_prefix_tokens = nullptr) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return false;
        out_root_node_id = it->second.root_node_id;
        out_total_tokens = it->second.total_tokens;
        if (out_prefix_tokens) {
            *out_prefix_tokens = it->second.prefix_tokens;
        }
        return true;
    }

    bool has_session(const std::string& session_id) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return sessions_.find(session_id) != sessions_.end();
    }

    bool remove_session(const std::string& session_id) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return sessions_.erase(session_id) > 0;
    }

    size_t count() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return sessions_.size();
    }

    bool save_to_disk(const std::string& path = "") const {
        std::string target = path.empty() ? registry_file_ : path;
        if (target.empty()) return false;

        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::ofstream ofs(target, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) return false;

        uint64_t magic = 0x5354524153455353ULL; // "STRASESS"
        uint32_t version = 1;
        uint32_t n = static_cast<uint32_t>(sessions_.size());
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));

        for (const auto& [id, rec] : sessions_) {
            uint32_t id_len = static_cast<uint32_t>(id.size());
            ofs.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
            ofs.write(id.data(), id_len);
            ofs.write(reinterpret_cast<const char*>(&rec.root_node_id), sizeof(rec.root_node_id));
            ofs.write(reinterpret_cast<const char*>(&rec.total_tokens), sizeof(rec.total_tokens));
            ofs.write(reinterpret_cast<const char*>(&rec.last_accessed_sec), sizeof(rec.last_accessed_sec));
            uint32_t n_tokens = static_cast<uint32_t>(rec.prefix_tokens.size());
            ofs.write(reinterpret_cast<const char*>(&n_tokens), sizeof(n_tokens));
            if (n_tokens > 0) {
                ofs.write(reinterpret_cast<const char*>(rec.prefix_tokens.data()),
                          n_tokens * sizeof(int32_t));
            }
        }
        return true;
    }

    bool load_from_disk(const std::string& path = "") {
        std::string target = path.empty() ? registry_file_ : path;
        if (target.empty()) return false;

        std::ifstream ifs(target, std::ios::binary);
        if (!ifs.is_open()) return false;

        uint64_t magic = 0;
        uint32_t version = 0;
        uint32_t n = 0;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        ifs.read(reinterpret_cast<char*>(&n), sizeof(n));

        if (magic != 0x5354524153455353ULL || version != 1) return false;

        std::unique_lock<std::shared_mutex> lock(mutex_);
        sessions_.clear();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t id_len = 0;
            ifs.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
            std::string id(id_len, '\0');
            ifs.read(&id[0], id_len);

            SessionRecord rec;
            rec.session_id = id;
            ifs.read(reinterpret_cast<char*>(&rec.root_node_id), sizeof(rec.root_node_id));
            ifs.read(reinterpret_cast<char*>(&rec.total_tokens), sizeof(rec.total_tokens));
            ifs.read(reinterpret_cast<char*>(&rec.last_accessed_sec), sizeof(rec.last_accessed_sec));
            uint32_t n_tokens = 0;
            ifs.read(reinterpret_cast<char*>(&n_tokens), sizeof(n_tokens));
            if (n_tokens > 0) {
                rec.prefix_tokens.resize(n_tokens);
                ifs.read(reinterpret_cast<char*>(rec.prefix_tokens.data()),
                         n_tokens * sizeof(int32_t));
            }
            sessions_[id] = std::move(rec);
        }
        return true;
    }

private:
    std::string registry_file_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, SessionRecord> sessions_;
};

}  // namespace strata::core
