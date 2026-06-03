#include <stdint.h>
#include <stdio.h>

int main() {
  int character = 'A';
  printf("%d\n", character);
  printf("%c %c %d %d %x %p\n", 65, 0x41, 65, 0x41, 0x41, (void *)0x41);
  char *str = "0";
  printf("%x %p\n", (long long int)str,
         (void *)str); // 为什么是：2625701b 0x645b2625701b ，前面少几个字节
  return 0;
}