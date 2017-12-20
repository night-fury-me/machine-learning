#include "library/headers.h"
#include "library/DecisionTree.h"

void SolveTask( int numberOfArgument, char *arguments[ ] ) {

        if( numberOfArgument != 5 ) {
                cout << "Incorrect number of arguments, please recheck !" << endl;
                return;
        }

        string trainingFile = arguments[ 1 ];
        string testDataFile = arguments[ 2 ];
        string trainigType  = arguments[ 3 ];
        string pruningThr   = arguments[ 4 ];
        stringstream ss( pruningThr );
        int pruningThreshold; ss >> pruningThreshold;

        if( trainigType  != "optimized" && trainigType != "randomized" && trainigType != "forest3" && trainigType != "forest15" ) {
                cout << "Training Type is not valid ! " << endl;
                return ;
        }

        DecisionTree dTree;
        dTree.SetPruningThreshold( pruningThreshold );
        dTree.Train( trainingFile, trainigType );
        dTree.PrintForest( "Forest.txt" );
        dTree.TestDataPrediction( testDataFile, "testResult.txt" );
}

int main( int argc, char *argv[ ] ) {

        SolveTask( argc, argv );

        return 0;
}
