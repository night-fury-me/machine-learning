#include "headers.h"

#define MIN             0
#define MAX             1

#define MEAN            0
#define SD              1

#define DISTANCE        0
#define CLASS           1
#define INDEX           2

#define int64           long long

class KNN {

        int k;
        vector< vector<double> > trainingData;
        vector< tuple<double,double> > extrems;

public:
        KNN( ) {  }
        KNN( int _k ) { k = _k; }

        void ClassifyObject( vector<double>& testObj, int objId,
                             double& classificationAccuracy, int& correctPrediction, ofstream& out ) {

                // Normalizing the test data.
                NormalizeData( testObj );

                // Storing the distances from the current test
                // data point to all training data points.
                int idx = 0;
                vector< tuple<double,int,int> > neighbors;

                for( auto& data: trainingData ) {
                        double dist = GetDistance( testObj, data );
                        neighbors.push_back( make_tuple( dist, int( data.back() ), idx ++ ) );
                }

                // Sorting on the basis of distance. to get nearest neighbors.
                std::sort( neighbors.begin(), neighbors.end(),
                       []( tuple<double,int,int> A, tuple<double,int,int> B )->bool {
                                return get<DISTANCE>( A ) < get<DISTANCE> ( B );
                } );

                /*
                cout << " -------------------------------------------------" << endl;
                for( int i = 0; i < 100; i ++ ) {
                        cout << get<INDEX> (neighbors[ i ]) << " ";
                } cout << endl;
                cout << "--------------------------------------------------" << endl;
                */

                unordered_map<int,int> vote, rmap;

                // These Stuffs are just used to make unordered_map faster, nothing to do with actual task.
                vote.max_load_factor( 0.25 );  vote.reserve( 1 << 10 );
                rmap.max_load_factor( 0.25 );  rmap.reserve( 1 << 10 );
                // --------------------------------------------------------------------------------------

                // Voting from the first k nearest neighbors.
                for( int i = 0; i < k; i ++ ) {
                        vote[ get<CLASS>( neighbors[ i ] ) ] ++;
                        rmap[ get<CLASS>( neighbors[ i ] ) ] = i;
                }


                // Selecting best candidate by vote count.
                int winner = 0, mxVote = 0;
                for( auto& x: vote ) {
                        if( mxVote < x.second ) {
                                winner = x.first;
                                mxVote = x.second;
                        }
                }

                vector<int> tie;
                for( auto& x: vote ) {
                        if( x.second == mxVote ) tie.push_back( x.first );
                }

                // In case of Tie, choosing a random candidate
                // from the list of tied candidates.
                srand( int( time( 0 ) ) );
                int id = rand( ) % int( tie.size( ) );

                winner = tie[ id ];
                double accuracy = 0.0;

                if( winner == testObj.back( ) )
                        accuracy += 1.0;
                else if( std::find( tie.begin(), tie.end(), int( testObj.back() ) ) != tie.end() )
                        accuracy += 1.0 / double( tie.size() );
                else    accuracy  = 0.0;

                classificationAccuracy += accuracy;
                correctPrediction      += accuracy > 0.0;

                int TrueClass   = int( testObj.back() );
                int NeighborIdx = get<INDEX>   ( neighbors[ rmap[ winner ] ] );
                double Distance = get<DISTANCE>( neighbors[ rmap[ winner ] ] );

                PrintTestResult( out, objId, winner, TrueClass, NeighborIdx, Distance, accuracy );
        }


        // Euclidean Distance Function.

        double GetDistance( vector<double> pointA, vector<double> pointB ) {
                double dist = 0.0;
                for( int i = 0; i < pointA.size()-1; i ++ ) {
                        dist += ( pointA[ i ] - pointB[ i ] ) * ( pointA[ i ] - pointB[ i ] );
                }
                return sqrt( dist );
        }


        void TestDataPrediction( string dataFile, string outputFile ) {

                ofstream out( outputFile );

                vector< vector<double> > testData;
                PrepareTestData( dataFile, testData );

                double classificationAccuracy = 0.0;
                int objId = 0, correctPrediction = 0;

                for( auto& testObj: testData ) {
                        ClassifyObject( testObj , objId ++, classificationAccuracy, correctPrediction, out );
                }

                out <<"classification accuracy=" << fixed << setprecision(4);
                out << double( classificationAccuracy / testData.size() ) << endl;
                cout <<"classification accuracy=" << fixed << setprecision(4);
                cout << double( classificationAccuracy / testData.size() ) << endl;

                int incorrectPrediction = int( testData.size( ) ) - correctPrediction;
                out << "Correct Predictions = " << correctPrediction << ", Incorrect Predictions = " << incorrectPrediction << endl;
                cout << "Correct Predictions = " << correctPrediction << ", Incorrect Predictions = " << incorrectPrediction << endl;
        }

        // Normalizing a data point with MEAN and STANDARD DEVIATION(SD)
        // Formula :    normalized_value = ( current_value - MEAN ) / STANDARD_DEVIATION

        void NormalizeData( vector<double>& data ) {
                for( int i = 0; i < data.size( )-1; i ++ ) {
                        tuple<double,double> curr = extrems[ i ];
                        data[ i ] = ( data[ i ] - get<MEAN> ( curr ) ) / get<SD> ( curr );
                }
        }

        void PrepareTrainingData( string dataFile  ) {
                ifstream inFile( dataFile );

                if( !inFile ) {
                        cout << "Training Data File Not Found!"  << endl;
                        return;
                }

                PrepareData( inFile, trainingData );

                for( int attr = 0; attr < trainingData[ 0 ].size( ); attr ++ ) {
                        // Calculating Mean.
                        double Mean = 0.0;
                        for( int row = 0; row < trainingData.size( ); row ++ ) {
                                Mean += trainingData[ row ][ attr ];
                        }
                        Mean /= double( trainingData.size() );

                        // Calculating Standard Deviation.
                        double standardDeviation = 0.0;
                        for( int row = 0; row < trainingData.size( ); row ++ ) {
                                standardDeviation += ( trainingData[ row ][ attr ] - Mean ) * ( trainingData[ row ][ attr ] - Mean );
                        }

                        standardDeviation /= double( trainingData.size( ) );
                        standardDeviation = sqrt( standardDeviation );

                        extrems.push_back( make_tuple( Mean, standardDeviation ) );
                }

                // Normalizing All Training Data points.
                for( auto& data: trainingData ) {
                        NormalizeData( data );
                }
        }

        void PrepareTestData( string dataFile, vector< vector<double> >& TestData ) {

                ifstream inFile( dataFile );

                if( !inFile ) {
                        cout << "Test Data File Not Found!"  << endl;
                        return;
                }

                PrepareData( inFile, TestData );
        }


        // Just A utility function for dumping
        // Training Data into a specified File .

        void PrintData( ofstream& out ) {
                for( auto& data: trainingData ) {
                        for( auto& x: data ) out << x << " ";
                        out << endl;
                }
        }

        void PrepareData( ifstream& inFile, vector< vector<double> >& examples ) {

                string rowData;

                while( getline( inFile, rowData ) ) {
                        stringstream inputStream( rowData );
                        double rowValue;
                        vector<double> rowValuesVector;
                        while( inputStream >> rowValue ) {
                                rowValuesVector.push_back( rowValue );
                        }
                        examples.push_back( rowValuesVector );
                }
        }

        void PrintTestResult( ofstream& out, int& id, int& Predicted, int True, int nnID, double dist, double& accuracy  ) {
                // Outputs in a file.
                out << "ID="        << setw(5)<<right<<id             <<", predicted="<<setw(3)      <<right          <<Predicted;
                out << ", true="    << setw(3)<<right<<True           <<", nn="       <<setw(5)      <<right          <<nnID;
                out << ", distance="<< setw(7)<<fixed<<setprecision(3)<<dist          <<", accuracy="<<setprecision(4)<<accuracy<<endl;

                // Outputs in console.
                cout << "ID="        << setw(5)<<right<<id             <<", predicted="<<setw(3)      <<right          <<Predicted;
                cout << ", true="    << setw(3)<<right<<True           <<", nn="       <<setw(5)      <<right          <<nnID;
                cout << ", distance="<< setw(7)<<fixed<<setprecision(3)<<dist          <<", accuracy="<<setprecision(4)<<accuracy<<endl;
        }
};
