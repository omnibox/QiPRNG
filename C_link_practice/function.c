
#include <stdio.h>


int NM = 0;

int myFunction(int num)
{
    if (num == 0)
 
        // if number is 0, do not perform any operation.
        return 0;
    else
        // if number is power of 2, return 1 else return 0
          return ((num & (num - 1)) == 0 ? 1 : 0) ;
 
}

void linarray(int* arr, int num)
{
    for(int i = 0; i < num; i++)
    {
        arr[i] = i * i;
    }
}

void setnum(int n)
{
	NM = n;
	printf("NUM is now %d\n", NM);
}

int getnum()
{
	printf("NUM is %d\n", NM);
	return NM;
}
