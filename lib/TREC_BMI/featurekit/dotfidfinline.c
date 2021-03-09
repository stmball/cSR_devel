// a 1 653668 ./rcv1v2/002/2286.vocab: 11

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int tok, df, tf;
//double N=804414; // rcv1
double N=567528;  // trec4
char doc[1000], prev[1000];
int n,T[100000];
double W[100000];

doit(){
   int i;
   //*strstr(prev,":") = 0;
   printf("%s",prev,".");
   double sum = 0;
   for (i=0;i<n;i++) {
      sum += W[i]*W[i];
   }
   sum = sqrt(sum);
   for (i=0;i<n;i++) {
      printf(" %d:%0.8lf",T[i],W[i]/sum);
   }
   printf("\n");
}

main(int argc, char **argv){
   if (argc > 1) N = atoi(argv[1]);
   while (4 == scanf("%*s %d %d %s %d",&tok,&df,doc,&tf)) {
      if (strcmp(doc,prev)) {
         if (*prev) doit();
         strcpy(prev,doc);
         n = 0;
      }
      if (df < 2) continue;
      double tfidf = (1 + log(tf)) * log(N/df);
      //printf("%d %d %d %0.4lf %s\n",tok,df,tf,tfidf,doc);
      T[n] = tok;
      W[n++] = tfidf;
   }
   doit();
}
