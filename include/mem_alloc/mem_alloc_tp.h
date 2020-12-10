#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <list>

#include <iterator>
#include <map>
#include <new>
#include <typeinfo>
using namespace std::chrono;
using namespace std;
#define DEBUG 0
unsigned long CALLOC_NUM = 2 * 2097152;
typedef char ALIGN[16];
#define COMMA ,
#define GETTYPE(ptr) (unsigned long long)ptr >> 48
#define ADDTYPE(ptr, type) \
    (void *)((unsigned long long)ptr | ((unsigned long long)type) << 48)
#define CLEANPTR(ptr, ptr_type) \
    ((ptr_type)((unsigned long long)ptr & 0xFFFFFFFFFFFFUL))

union header {
    struct {
        size_t size;
        unsigned is_free;
        union header *next;
        // char ALIGN[8];
    } s;
    /* force the header to be aligned to 16 bytes */
    ALIGN stub;
};
typedef union header header_t;
class obj_alloc;
class mem_alloc {
    unsigned long long total_size;
    header_t *head;
    header_t *tail;
    unsigned is_free;
    unsigned remaining_size;

  public:
    mem_alloc(unsigned long long _total_size) {
        cudaError_t err = cudaSuccess;
        void *block;
        total_size = _total_size;
        cudaMallocManaged(&block, _total_size);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n",
                    cudaGetErrorString(err));
            return;
        }
        is_free = 1;
        remaining_size = total_size - sizeof(header_t);
        head = tail = (header_t *)block;
        head->s.size = remaining_size;
        head->s.is_free = 1;
        head->s.next = NULL;
    }
    header_t *get_free_block(size_t size) {
        header_t *curr = tail;
        while (curr) {
            /* see if there's a free block that can accomodate requested size */
            if (curr->s.is_free && curr->s.size >= size) return curr;
            curr = curr->s.next;
        }
        printf("searching from head\n");
        curr = head;
        while (curr) {
            /* see if there's a free block that can accomodate requested size */
            if (curr->s.is_free && curr->s.size >= size) return curr;
            curr = curr->s.next;
        }
        return NULL;
    }
    void alloc_free_block(size_t size, header_t *block_ptr) {
        header_t *next_block;
        block_ptr->s.is_free = 0;
        next_block =
            (header_t *)(((char *)block_ptr) + size + sizeof(header_t));
        // printf(nex)
        block_ptr->s.is_free = 0;
        next_block->s.is_free = 1;
        next_block->s.next = block_ptr->s.next;
        next_block->s.size = block_ptr->s.size - size - sizeof(header_t);
        block_ptr->s.size = size;
        block_ptr->s.next = next_block;
        // printf("head %p nex %p  %d \n",block_ptr, next_block ,
        // sizeof(header_t));
        if (tail == block_ptr) tail = next_block;
        remaining_size = remaining_size - size - sizeof(header_t);
    }
    void custom_free(void *block) {
        header_t *header;
        /* program break is the end of the process's data segment */

        if (!block) return;
        header = (header_t *)((char *)block - sizeof(header_t));

        header->s.is_free = 1;
    }

    void *custom_malloc(size_t size) {
        // size_t total_size;

        header_t *header;

        if (!size) return NULL;
        // printf("BMalloc .....   ... TOTAL SIZE %x\n", size);
        size = 16 * floor(((size + 15)) / 16);
        header = get_free_block(size);
        if (header) {
            /* Woah, found a free block to accomodate requested memory. */
            alloc_free_block(size, header);
            // printf("malloc:%p\n",(void *)(header + sizeof(header_t)));
            //  printf("AMalloc .....   ... %p %x\n",
            //         (void *)((char *)header + sizeof(header_t)),
            //         header->s.size);

            return (void *)((char *)header + sizeof(header_t));
        }
        printf("RETNULL");
        return NULL;
    }

    template <class myType>
    void *calloc(int count) {
        void *ptr = custom_malloc(sizeof(myType) * count);
        // printf("%s ..... %p  ... TOTAL SIZE %x\n", typeid(myType).name(),
        // ptr,
        //        sizeof(myType) * count);
        return ptr;
    }
    template <class myType>
    bool realloc(void *ptr, int old_count, int new_count) {
        header_t *header;
        header_t *next_header;
        header_t *next_next_header;
        unsigned long size_diff = sizeof(myType) * (new_count - old_count);

        header = (header_t *)((char *)(ptr) - sizeof(header_t));
        next_header = header->s.next;
        if (!next_header->s.is_free || next_header->s.size < size_diff) {
            printf("REALLOC FAILD %d , %d %d %d\n", old_count, new_count,
                   next_header->s.is_free, next_header->s.size);
            return false;
        }

        next_next_header = (header_t *)((char *)next_header + size_diff);
        next_next_header->s.is_free = 1;
        next_next_header->s.size = next_header->s.size - size_diff;
        next_next_header->s.next = next_header->s.next;
        if (tail == next_header) tail = next_next_header;
        header->s.size += size_diff;
        header->s.next = next_next_header;
        printf("REALLOC Sucs %d , %d \n", old_count, new_count);
        return true;
    }
};

class range_bucket {
  public:
    unsigned count;
    unsigned type_size;
    void *mem_ptr;
    unsigned total_mem_bytes;
    unsigned total_count;

    range_bucket(unsigned _total, unsigned _type_size, void *_ptr) {
        count = 0;
        type_size = _type_size;
        mem_ptr = _ptr;
        total_count = _total;  // floor(_total / type_size);
        printf("total_count %d , total %d , type_size %d \n", total_count,
               _total, type_size);
    }
    void *get_next_mem(unsigned num_of_obj = 1) {
        void *ptr = (void *)((char *)mem_ptr + type_size * count);
        count += num_of_obj;
        if (DEBUG) printf("count:%u\n", count);
        return ptr;
    }

    void *get_range_start() { return mem_ptr; }
    void *get_range_end() {
        return (void *)((char *)mem_ptr + type_size * total_count);
    }
    bool is_contiguous_chunk(void *new_mem_ptr) {
        printf(
            "extend old_mem : %p , new_mem : %p, diff: %p , total_size : "
            "%x "
            "total_size_withH : %x expect : %p diff %p \n",
            mem_ptr, new_mem_ptr, (char *)new_mem_ptr - (char *)mem_ptr,
            type_size * (total_count),
            type_size * (total_count) + sizeof(header_t),
            (char *)mem_ptr + type_size * (total_count) + sizeof(header_t),
            (char *)new_mem_ptr - (char *)(mem_ptr + type_size * (total_count) +
                                           sizeof(header_t)));
        if (((char *)new_mem_ptr -
             (char *)(mem_ptr + type_size * (total_count) + sizeof(header_t))) %
                sizeof(header_t) ==
            0)
            return true;

        return false;
    }
    bool extend_mem_chunk(void *new_mem_ptr, unsigned new_total) {
        if (!is_contiguous_chunk(new_mem_ptr)) return false;
        total_count += new_total;
        printf("FUCKK\n");
        return true;
    }
    bool is_full() {
        if (DEBUG)
            printf("count:%u %u %d \n", count, total_count,
                   count == total_count);
        return count == total_count;
    }
    bool is_enough(unsigned num_of_obj) {
        return (total_count - count >= num_of_obj);
    }
};

#define FUNC_LEN 30
class obj_info_tuble {
  public:
    void *func[FUNC_LEN];
};

__managed__ __align__(16) char buf5[128];
template <class myType>
__global__ void dump_vtable(void **vtable, void **gpu_ptr) {
    // int tid = threadIdx.x;
    int i;
    myType *obj2;
    obj2 = new (buf5) myType();
    // // printf("dump\n");
    memcpy(gpu_ptr, obj2, sizeof(void *));
    long ***mVtable = (long ***)&obj2;
    // printf("kernal %p-----%p----------ptr
    // %p-\n",mVtable[0][0],mVtable[0],gpu_ptr);
    // void **mVtable = (void **)*vfptr;
    for (i = 0; i < FUNC_LEN; i++) {
        if (mVtable[0][0][i] == 0) break;
        vtable[i] = (void *)mVtable[0][0][i];
        // printf("kernal i :%d %p----------------\n", i, mVtable[0][0][i]);
    }
}
unsigned long INIT_CHUNK_SIZE = 512 * 1024;
class TypeContainer {
  public:
    list<range_bucket *> *type_bucket_list = NULL;
    void **vtable;
    unsigned range_size;
    unsigned typeSize;
    static const unsigned long MAX_CHUNK_SIZE = 1024 * 1024 * 1024;
    mem_alloc *mem;
    obj_alloc *obj_alloctor;
    void *gpu_vptr;
    void *cpu_vptr;
    unsigned long num_of_objs;
    unsigned long long type;

    TypeContainer(mem_alloc *_mem, obj_alloc *_obj_alloc) {
        this->type_bucket_list = new list<range_bucket *>();
        this->mem = _mem;
        this->obj_alloctor = _obj_alloc;
        this->vtable = (void **)mem->calloc<void *>(FUNC_LEN);
        INIT_CHUNK_SIZE = CALLOC_NUM;

        this->range_size = INIT_CHUNK_SIZE;
        this->num_of_objs = 0;
    }
    unsigned get_range_size() { return this->range_size; }
    void inc_chunk_size() {
        if (this->range_size < MAX_CHUNK_SIZE)
            this->range_size = 2 * this->range_size;
    }
    template <class myType>
    static TypeContainer *create(mem_alloc *_mem, obj_alloc *_obj_alloc) {
        TypeContainer *ptr = new TypeContainer(_mem, _obj_alloc);
        ptr->fill_vptr<myType>();
        range_bucket *bucket =
            new range_bucket(INIT_CHUNK_SIZE, sizeof(myType),
                             _mem->calloc<myType>(INIT_CHUNK_SIZE));
        ptr->type_bucket_list->push_front(bucket);
        ptr->typeSize = sizeof(myType);
        _obj_alloc->inc_num_of_ranges();

        return ptr;
    }
    template <class myType>
    void fill_vptr() {
        cudaError_t err = cudaSuccess;
        myType *temp_obj = new myType();
        memcpy(&this->cpu_vptr, temp_obj, sizeof(void *));
        free(temp_obj);

        void **gpu_vptr_temp = NULL;
        gpu_vptr_temp = (void **)mem->calloc<void *>(1);

        dump_vtable<myType><<<1, 1>>>(this->vtable, gpu_vptr_temp);
        cudaDeviceSynchronize();
        this->gpu_vptr = gpu_vptr_temp;
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: my_new failed (%s)\n",
                    cudaGetErrorString(err));
        }
        if (1)
            for (int ii = 0; ii < FUNC_LEN; ii++) {
                if (vtable[ii] == NULL) break;
                printf("vtable [%s][%d]:%p\n", typeid(myType).name(), ii,
                       vtable[ii]);
            }
    }
    template <class myType>
    range_bucket *add_new_bucket(void *mem_chunk_ptr) {
        range_bucket *bucket =
            new range_bucket(this->range_size, sizeof(myType), mem_chunk_ptr);

        type_bucket_list->push_front(bucket);

        obj_alloctor->inc_num_of_ranges();
        if (1)

            printf("range [%s] mem: %p\n", typeid(myType).name(),
                   bucket->mem_ptr);
        return bucket;
    }

    template <class myType>
    range_bucket *get_type_bucket() {
        range_bucket *bucket = NULL;
        bucket = type_bucket_list->front();

        if (bucket->is_full()) {
            this->inc_chunk_size();

            if (!this->mem->realloc<myType>(
                    bucket->mem_ptr, bucket->total_count,
                    bucket->total_count + this->range_size)) {
                void *new_mem_chunk_ptr =
                    this->mem->calloc<myType>(this->range_size);
                bucket = add_new_bucket<myType>(new_mem_chunk_ptr);
                return bucket;
            }
            bucket->total_count = bucket->total_count + this->range_size;
        }

        return bucket;
    }
    template <class myType>
    void *get_next_mem() {
        this->num_of_objs++;
        return get_type_bucket<myType>()->get_next_mem();
    }
};
typedef std::map<uint32_t, TypeContainer *> MAP;

__global__ void vptrPatch(void *array, void *vPtr, unsigned sizeofType, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) memcpy((char *)array + tid * sizeofType, vPtr, sizeof(void *));
}
class range_tree_node {
  public:
    void *range_start;
    void *range_end;
    void *mid;
    obj_info_tuble *tuble;
    int size;
    int size2;
    void set_range(void *start, void *end) {
        range_start = start;
        range_end = end;
        unsigned long long _start, _end;
        memcpy(&_start, &range_start, sizeof(void *));
        memcpy(&_end, &range_end, sizeof(void *));
        mid = (void *)((unsigned long long)((_start + _end) / 2));
    }
};
class obj_alloc {
    mem_alloc *mem;
    MAP type_map;
    unsigned num_of_ranges;
    obj_info_tuble *table;
    range_tree_node *range_tree;
    unsigned tree_size;

  public:
    ~obj_alloc() {
        MAP::iterator it;
        TypeContainer *type;
        unsigned i;
        unsigned long avg_size = 0;
        unsigned long total_obj = 0;
        for (it = type_map.begin(), i = 0; it != type_map.end(); i++, it++) {
            type = it->second;
            fprintf(stderr, "Type#%d:\n", i);
            fprintf(stderr,
                    "Type Size: %d \t Number of Buckets : %d \t Range Size: %d "
                    "\t Number of Objs : %d\n\n",
                    type->typeSize, type->type_bucket_list->size(),
                    type->range_size, type->num_of_objs);
            total_obj += type->num_of_objs;
            avg_size += type->typeSize * type->num_of_objs;
        }
        fprintf(stderr, "Avg Types Size %f\n", ((float)avg_size) / total_obj);
    }
    obj_alloc(mem_alloc *_mem, unsigned long num = 512 * 1024) {
        mem = _mem;
        num_of_ranges = 0;
        table = NULL;
        range_tree = NULL;
        CALLOC_NUM = num;
    }
    range_tree_node *get_range_tree() { return range_tree; }
    obj_info_tuble *get_vfun_table() { return this->table; }
    bool is_new_type(uint32_t hash) {
        return type_map.find(hash) == type_map.end();
    }
    unsigned get_tree_size() { return tree_size; }
    void inc_num_of_ranges() { num_of_ranges++; }
    inline uint32_t hash_str_uint32(const char *str) {
        uint32_t hash = 0x811c9dc5;
        uint32_t prime = 0x1000193;

        for (int i = 0; str[i] != '\0'; ++i) {
            uint8_t value = str[i];
            hash = hash ^ value;
            hash *= prime;
        }

        return hash;
    }
    template <class myType>
    void *my_new() {
        uint32_t hash = hash_str_uint32(typeid(myType).name());
        if (is_new_type(hash)) {
            // not found
            if (DEBUG)
                printf("class was not FOUND %s ---\n", typeid(myType).name());
            type_map[hash] = TypeContainer::create<myType>(this->mem, this);
            type_map[hash]->type = type_map.size() ;

        } else {
            // found
            if (DEBUG) printf("class FOUND %p \n", type_map[hash]);
        }

        // we have the bucket with space
        return ADDTYPE(type_map[hash]->get_next_mem<myType>(),
                       type_map[hash]->type);
    }
    template <class myType>
    void *calloc(unsigned num) {
        return (void *)mem->calloc<myType>(num);
    }

    // unsigned get_type_tubles_frm_list(obj_info_tuble *table,
    //                                   TypeContainer *type) {
    //     list<range_bucket *>::iterator iter;
    //     int i;
    //     for (iter = type->type_bucket_list->begin(), i = 0;
    //          iter != type->type_bucket_list->end(); ++iter, i++) {
    //         table[i].range_start = (*iter)->get_range_start();
    //         table[i].range_end = (*iter)->get_range_end();

    //         memcpy(&(table[i].func[0]), &type->vtable[0],
    //                sizeof(void *) * FUNC_LEN);
    //     }
    //     return i;
    // }

    unsigned create_table(obj_info_tuble *table) {
        MAP::iterator it;
        unsigned i;
        for (it = type_map.begin(), i = 0; it != type_map.end(); it++) {
            memcpy(&(table[it->second->type].func[0]), &it->second->vtable[0],
                   sizeof(void *) * FUNC_LEN);
        }
        return i;
    }
    void create_table() {
        this->table =
            (obj_info_tuble *)mem->calloc<obj_info_tuble>(type_map.size()+1);
        create_table(this->table);

        for (int i = 0; i < this->type_map.size()+1; i++) {
            printf("Type%d:  %p   \n", i, this->table[i].func[0]);
        }
        printf("#############\n");
    }

    // void sort_table() {
    //     int min;
    //     int size = this->num_of_ranges;
    //     obj_info_tuble temp;
    //     for (int i = 0; i < size; i++) {
    //         min = i;

    //         for (int j = i + 1; j < size; j++) {
    //             if (table[min].range_start > table[j].range_start) {
    //                 min = j;
    //             }
    //         }

    //         memcpy(&temp, &table[i], sizeof(obj_info_tuble));
    //         memcpy(&table[i], &table[min], sizeof(obj_info_tuble));
    //         memcpy(&table[min], &temp, sizeof(obj_info_tuble));
    //     }
    // }

    // void create_tree(int level, unsigned start, unsigned end) {
    //     if (level == -1) return;

    //     // printf("startt %d end %d level %d %\n", start, end, level);
    //     for (int i = start; i < end; i++) {
    //         if (this->range_tree[2 * i + 2].range_end)
    //             this->range_tree[i].set_range(
    //                 this->range_tree[2 * i + 1].range_start,
    //                 this->range_tree[2 * i + 2].range_end);
    //         else
    //             this->range_tree[i].set_range(
    //                 this->range_tree[2 * i + 1].range_start,
    //                 this->range_tree[2 * i + 1].range_end);
    //         //   this->range_tree[unsigned((i - 1) / 2)].set_range(
    //         //       this->range_tree[i].range_start,
    //         //       this->range_tree[i].range_end);
    //     }
    //     create_tree(level - 1, (1 << (level - 1)) - 1, start);
    // }
    // void create_tree() {
    //     unsigned tree_depth = (unsigned)ceil(log2(num_of_ranges));
    //     unsigned power2 = ((1 << (tree_depth)));
    //     unsigned level = tree_depth;
    //     unsigned tree_alloc_size = ((1 << (tree_depth + 1))) - 1;
    //     this->tree_size = power2 + num_of_ranges - 1;
    //     this->range_tree =
    //         (range_tree_node *)mem->calloc<range_tree_node>(tree_alloc_size);

    //     if (1)
    //         printf("tree %d number or ranges %d   alloc_size %d\n", tree_size,
    //                num_of_ranges, tree_alloc_size);
    //     create_table();
    //     // for (int i = 0; i < this->num_of_ranges; i++) {
    //     //     for (int k = 0; k < 10; k++)
    //     //         printf("1- %d: %p %p   \n", i, this->table[i].range_start,
    //     //                this->table[i].func[k]);
    //     // }
    //     sort_table();
    //     // for (int i = 0; i < this->num_of_ranges; i++) {
    //     //     for (int k = 0; k < 10; k++)
    //     //         printf("2- %d: %p %p   \n", i, this->table[i].range_start,
    //     //                this->table[i].func[k]);
    //     // }
    //     range_tree[tree_size - 1].size = power2 - 1;
    //     range_tree[tree_size - 1].size2 = power2 + num_of_ranges - 1;
    //     int j = 0;
    //     // for (int i = 0; i < this->num_of_ranges; i++) {
    //     //     for (int k = 0; k < 4; k++)
    //     //         printf("b3- %d: %p %p   \n", i, this->table[i].range_start,
    //     //                this->table[i].func[k]);
    //     // }
    //     for (int i = power2 - 1; i < power2 + num_of_ranges - 1; j++, i++) {
    //         // printf("%d\n", i);
    //         // for (int i = 0; i < this->num_of_ranges; i++) {
    //         //     for (int k = 0; k < 4; k++)
    //         //         printf("b3- %d: %p %p   \n", i,
    //         //         this->table[i].range_start,
    //         //                this->table[i].func[k]);
    //         // }
    //         range_tree[i].set_range(this->table[j].range_start,
    //                                 this->table[j].range_end);
    //         range_tree[i].tuble = &(this->table[j]);
    //         // for (int i = 0; i < this->num_of_ranges; i++) {
    //         //     for (int k = 0; k < 4; k++)
    //         //         printf("a3- %d: %p %p   \n", i,
    //         //         this->table[i].range_start,
    //         //                this->table[i].func[k]);
    //         // }
    //     }

    //     int mm = power2 + num_of_ranges - 2;
    //     unsigned sizeoflast = ((char *)(range_tree[mm].range_end) -
    //                            (char *)(range_tree[mm].range_start));
    //     for (int i = power2 + num_of_ranges - 1; i < tree_size; j++, i++) {
    //         // printf("%d\n", i);
    //         range_tree[i].set_range(
    //             range_tree[i - 1].range_end,
    //             (char *)range_tree[i - 1].range_end + sizeoflast);
    //         range_tree[i].mid = 0;
    //         range_tree[i].tuble = (0);
    //     }
    //     // for (int i = 0; i < this->num_of_ranges; i++) {
    //     //     for (int k = 0; k < 10; k++) {
    //     //         printf("33%d: %p %p   \n", i, this->table[i].range_start,
    //     //                this->table[i].func[k]);
    //     //     }
    //     // }
    //     j = 0;
    //     // for (int i = power2 - 1; i < power2 + num_of_ranges - 1; j++, i++) {
    //     //     printf("%d\n", i);

    //     //     printf("%p ---------eqwewq-\n", range_tree[i].tuble->func[0]);
    //     // }
    //     create_tree(level - 1, (1 << (level - 1)) - 1,
    //                 (1 << (level - 1)) - 1 + (1 << (level - 1)));
    //     if (1)
    //         for (int i = 0; i < tree_size; i++) {
    //             printf("%d: %p %p %p  \n", i, this->range_tree[i].range_start,
    //                    this->range_tree[i].range_end, this->range_tree[i].mid);
    //             // obj_info_tuble *tuble = this->range_tree[i].tuble;
    //             // void **vtable;
    //             // if (0) {
    //             //   vtable = &this->range_tree[i].tuble->func[0];
    //             //   for (int ii = 0; ii < FUNC_LEN; ii++) {
    //             //     printf("vtable[%d]:%p\n", ii, vtable[ii]);
    //             //   }
    //             // }
    //         }
    // }
    __host__ __device__ void **get_vfunc(void *obj) {
        unsigned ptr = 0;
        unsigned next_ptr = 0;
        while (true) {
            if (obj > range_tree[ptr].mid)
                next_ptr = 2 * ptr + 1;

            else
                next_ptr = 2 * ptr + 2;

            if (next_ptr >= tree_size) return &(range_tree[ptr].tuble->func[0]);
            if (DEBUG)
                printf("mid %p %d %d tree : %d \n", range_tree[ptr].mid, ptr,
                       next_ptr, tree_size);
            ptr = next_ptr;
        }
    }

    void type_vptr_patch(list<range_bucket *> *list_ptr, void *vptr) {
        list<range_bucket *>::iterator iter;
        int block_size = 256;
        int num_blocks = (CALLOC_NUM + block_size - 1) / block_size;

        for (iter = list_ptr->begin(); iter != list_ptr->end(); ++iter) {
            num_blocks = ((*iter)->total_count + block_size - 1) / block_size;
            dim3 threads(block_size, 1, 1);
            dim3 grid(num_blocks, 1, 1);
            vptrPatch<<<grid, threads>>>((*iter)->mem_ptr, vptr,
                                         (*iter)->type_size,
                                         (*iter)->total_count);
            cudaDeviceSynchronize();
        }
    }
    void toDevice() {
        MAP::iterator it;
        for (it = type_map.begin(); it != type_map.end(); it++) {
            type_vptr_patch(it->second->type_bucket_list, it->second->gpu_vptr);
            printf("gpuptr %p \n", it->second->gpu_vptr);
        }
    }
    void toHost() {
        MAP::iterator it;
        for (it = type_map.begin(); it != type_map.end(); it++) {
            type_vptr_patch(it->second->type_bucket_list, it->second->cpu_vptr);
        }
    }
};

__host__ __device__ bool inRange(void *obj, range_tree_node *range_tree,
                                 unsigned ptr) {
    if (1) {
        return obj >= range_tree[ptr].range_start &&
               obj <= range_tree[ptr].range_end;
    }
    return false;
}

__host__ __device__ void **get_vfunc_tree(void *obj,
                                          range_tree_node *range_tree,
                                          unsigned tree_size) {
    unsigned ptr = 0;
    unsigned next_ptr = 0;
    while (true) {
        // printf("looking %p ----- %d %d\n", obj, ptr , next_ptr);
        if (obj < range_tree[ptr].mid)
            next_ptr = 2 * ptr + 1;

        else
            next_ptr = 2 * ptr + 2;

        if (next_ptr >= tree_size) {
            // printf("Found %p ----- %d %d\n", obj, ptr, next_ptr);
            return &(range_tree[ptr].tuble->func[0]);
        }
        if (range_tree[next_ptr].mid == 0) next_ptr = 2 * ptr + 1;
        if (DEBUG)
            printf("mid %p %d %d tree : %d \n", range_tree[ptr].mid, ptr,
                   next_ptr, tree_size);
        ptr = next_ptr;
    }
}

__host__ __device__ void **get_vfunc_tree_2(void *obj,
                                            range_tree_node *range_tree,
                                            unsigned tree_size) {
    unsigned ptr = 0;
    unsigned next_ptr = 0;
    while (true) {
        // printf("looking %p ----- %d %d\n", obj, ptr , next_ptr);
        next_ptr = 2 * ptr + 1;
        if (next_ptr >= tree_size) {
            // printf("Found %p ----- %d %d\n", obj, ptr, next_ptr);
            return &(range_tree[ptr].tuble->func[0]);
        }
        if (!inRange(obj, range_tree, 2 * ptr + 1)) next_ptr = 2 * ptr + 2;

        if (DEBUG)
            printf("mid %p %d %d tree : %d \n", range_tree[ptr].mid, ptr,
                   next_ptr, tree_size);
        ptr = next_ptr;
    }
}

__device__ void **get_vfunc_type(void *obj, obj_info_tuble *vfun_table) {
    unsigned long type;

    type = GETTYPE(obj);
    return &(vfun_table[type].func[0]);


}

__device__ void **get_vfunc_itr(void *obj, range_tree_node *range_tree,
                                unsigned tree_size) {
    int idx = range_tree[tree_size - 1].size;
    int lim = range_tree[tree_size - 1].size2;

    // assert(idx>2);
    for (int i = idx; i < lim; i++) {
        if (inRange(obj, range_tree, i)) {
            // if (tid == 0)
            //     printf("Found %p ----- %d  %p %p\n", obj, i,
            //            &range_tree[i].tuble->func[0],
            //            range_tree[i].tuble->func[2]);
            assert(range_tree[i].mid != NULL);

            return &(range_tree[i].tuble->func[0]);
        }
    }
    // printf("not Found %p ----- \n", obj);
    assert(false);
    return NULL;
}
__device__ void **get_vfunc(void *obj, range_tree_node *range_tree,
                            unsigned tree_size) {
    return get_vfunc_tree_2(obj, range_tree, tree_size);
    return get_vfunc_itr(obj, range_tree, tree_size);
}