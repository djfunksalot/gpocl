/*
   
   Structure of a genome (individual)

     |                         |
+----+-----+----+--------------+-------------
|size|arity|type| index/value  |  ...
| 32 |  3  | 7  |     22       |
+----+-----+----+--------------+-------------
     |    first element        | second ...

*/
#define COMPACT_RANGE 4194303 // 2^22 - 1;
#define SCALE_FACTOR 1024

#define ARITY( packed ) ((packed & 0xE0000000) >> 29) // 0xE0000000 = 11100000 00000000 00000000 00000000
#define INDEX( packed ) ((packed & 0x1FC00000) >> 22) // 0x1FC00000 = 00011111 11000000 00000000 00000000
#define AS_INT( packed ) (packed & 0x3FFFFF)          // 0x3FFFFF = 00000000 00111111 11111111 11111111
#define AS_FLOAT( packed ) ((float)( packed & 0x3FFFFF ) * SCALE_FACTOR / COMPACT_RANGE) // 0x3FFFFF = 00000000 00111111 11111111 11111111
