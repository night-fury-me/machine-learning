#include "library/headers.h"
#include "library/KNN.h"

void SolveTask( int numberOfArgument, char *arguments[ ] ) {
        if( numberOfArgument != 5 ) {
                cout << "Invalid number of arguments. " << endl;
                return;
        }

        string trainingData = arguments[ 1 ];
        string testData     = arguments[ 2 ];
        string k            = arguments[ 4 ];

        stringstream ss( k );
        int valueK; ss >> valueK;

        KNN knn( valueK );
        knn.PrepareTrainingData( trainingData );
        knn.TestDataPrediction ( testData, "predictions.txt" );
}

int main( int argc, char *argv[ ] ) {

        SolveTask( argc, argv );

        return 0;
}

