#include "graph.h"

TopGraph g;

int main(int argc, char ** argv){
  g.init();
  g.run(10);
  g.end();
  return 0;
}
