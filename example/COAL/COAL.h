
__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size;
__managed__ void *temp_coal;

#define COAL_S1_inc(ptr){   vtable = get_vfunc(ptr, range_tree, tree_size);  temp_coal = vtable[0]; }
#define COAL_S1_dec(ptr){   vtable = get_vfunc(ptr, range_tree, tree_size);  temp_coal = vtable[1]; }
