__managed__ obj_info_tuble *vfun_table; // to hold a pointer to vtable
__managed__ void *temp_TP; // used by the TP vfuns call macros

#define TP_S1_inc(ptr)                                                         \
  {                                                                            \
    vtable = get_vfunc_type(ptr, vfun_table);                                  \
    temp_TP = vtable[0];                                                       \
  }
#define TP_S1_dec(ptr)                                                         \
  {                                                                            \
    vtable = get_vfunc_type(ptr, vfun_table);                                  \
    temp_TP = vtable[1];                                                       \
  }
