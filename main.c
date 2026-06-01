#include <ctype.h>
#include <stdio.h>

int main() {
  for (int i = 0; i < 256; ++i) {
    if (isprint(i))
      printf("%3d -> '%c'\n", i, i);
    else
      printf("%3d -> (non-printable)\n", i);
  }
  return 0;
}