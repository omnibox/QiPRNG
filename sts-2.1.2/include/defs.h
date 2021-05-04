/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                       D E B U G G I N G  A I D E S
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "config.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                              M A C R O S
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define MAX(x,y)             ((x) <  (y)  ? (y)  : (x))
#define MIN(x,y)             ((x) >  (y)  ? (y)  : (x))
#define isNonPositive(x)     ((x) <= 0.e0 ?   1  : 0)
#define isPositive(x)        ((x) >  0.e0 ?   1 : 0)
#define isNegative(x)        ((x) <  0.e0 ?   1 : 0)
#define isGreaterThanOne(x)  ((x) >  1.e0 ?   1 : 0)
#define isZero(x)            ((x) == 0.e0 ?   1 : 0)
#define isOne(x)             ((x) == 1.e0 ?   1 : 0)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
                         G L O B A L  C O N S T A N T S
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define ALPHA							0.01	/* SIGNIFICANCE LEVEL */
#define MAXNUMOFTEMPLATES				148		/* APERIODIC TEMPLATES: 148=>temp_length=9 */
#define NUMOFTESTS						15		/* MAX TESTS DEFINED  */
#define NUMOFPVALS						188		/* MAX PVALS DEFINED  */
#define NUMOFGENERATORS					10		/* MAX PRNGs */
#define MAXFILESPERMITTEDFORPARTITION	148
#define	TEST_FREQUENCY					1
#define	TEST_BLOCK_FREQUENCY			2
#define	TEST_CUSUM						3
#define	TEST_RUNS						4
#define	TEST_LONGEST_RUN				5
#define	TEST_RANK						6
#define	TEST_FFT						7
#define	TEST_NONPERIODIC				8
#define	TEST_OVERLAPPING				9
#define	TEST_UNIVERSAL					10
#define	TEST_APEN						11
#define	TEST_RND_EXCURSION				12
#define	TEST_RND_EXCURSION_VAR			13
#define	TEST_SERIAL						14
#define	TEST_LINEARCOMPLEXITY			15
#define	PVAL_FREQUENCY					0
#define	PVAL_BLOCK_FREQUENCY			1
#define	PVAL_CUSUM_F					2
#define	PVAL_CUSUM_R					3
#define	PVAL_RUNS						4
#define	PVAL_LONGEST_RUN				5
#define	PVAL_RANK						6
#define	PVAL_FFT						7
#define	PVAL_NONPERIODIC				8
#define	PVAL_OVERLAPPING				156
#define	PVAL_UNIVERSAL					157
#define	PVAL_APEN						158
#define	PVAL_RND_EXCURSION				159
#define	PVAL_RND_EXCURSION_VAR			167
#define	PVAL_SERIAL						185
#define	PVAL_LINEARCOMPLEXITY			187


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                   G L O B A L   D A T A  S T R U C T U R E S
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

typedef unsigned char	BitSequence;

typedef struct _testParameters {
	int		n;
	int		blockFrequencyBlockLength;
	int		nonOverlappingTemplateBlockLength;
	int		overlappingTemplateBlockLength;
	int		serialBlockLength;
	int		linearComplexitySequenceLength;
	int		approximateEntropyBlockLength;
	int		numOfBitStreams;
} TP;
