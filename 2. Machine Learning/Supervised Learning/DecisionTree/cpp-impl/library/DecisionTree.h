#include "headers.h"
#include "BigInteger.h"

#define MIN             0
#define MAX             1

#define THRESHOLD       1
#define GAIN            2
#define ATTRIBUTE       0

#define int64           long long

class Node {

public:
        bool isLeaf, inited;
        int attribute; bigint nodeId;
        double gain, threshold;
        unordered_map<int,double> distribution;

        Node *Left, *Right;

        Node( ) { }
        Node( bool _isLeaf, unordered_map<int,double> _distribution, bigint _nodeId  ) {
                inited = true;
                nodeId = _nodeId;
                isLeaf = _isLeaf;
                Left = Right = NULL;
                distribution = _distribution;
        }
        Node( double _attribute, double _threshold, double _gain, bool _isLeaf, bigint _nodeId ) {
                inited = true;
                nodeId = _nodeId;
                Left = Right = NULL;
                isLeaf = _isLeaf, gain = _gain;
                attribute = _attribute, threshold = _threshold;
        }
        void Extend( ) {
                if( Left  == NULL ) Left  = new Node( );
                if( Right == NULL ) Right = new Node( );
        }
};

class DecisionTree {

        vector<Node*> Forest;
        int PruningThreshold;
public:
        DecisionTree( ) {  }

        void SetForestSize( int TreeCount ) {
                Forest.resize(  TreeCount );
        }

        void SetPruningThreshold( int threshold  ) {
                this->PruningThreshold = threshold;
        }

        int GetPruningThreshold( ) {
                return this->PruningThreshold;
        }

        double InformationGain( vector< vector<double> >& examples, int attribute, double threshold ) {

                int numberOfExamples = examples.size();
                int numberOfColumns  = examples[ 0 ].size();
                vector< vector<double> > leftChild, rightChild;

                for( auto& x: examples ) {
                        if( x[ attribute ] < threshold ) leftChild.push_back( x );
                        else                             rightChild.push_back( x );
                }

                int numberOfExampleInLeft  = leftChild.size( );
                int numberOfExampleInRight = rightChild.size( );

                unordered_map<int,int> MainNodeFreq, LeftNodeFreq, RightNodeFreq;

                /// These Stuffs are just used to make unordered_map faster, nothing to do with actual task.
                MainNodeFreq.max_load_factor( 0.25 );   MainNodeFreq.reserve( 1 << 10 );
                RightNodeFreq.max_load_factor( 0.25 );  RightNodeFreq.reserve( 1 << 10 );
                LeftNodeFreq.max_load_factor( 0.25 );   LeftNodeFreq.reserve( 1 << 10 );
                /// --------------------------------------------------------------------------------------

                auto FrequecyCount = [&]( vector< vector<double> >& vec, unordered_map<int,int>& FreqArr )->void {
                        for( auto& x: vec ) FreqArr[ int( x.back( ) ) ] ++;
                };

                FrequecyCount( examples,   MainNodeFreq  );
                FrequecyCount( leftChild,  LeftNodeFreq  );
                FrequecyCount( rightChild, RightNodeFreq );

                /// Total Information Gain, IG = H - ( HL * PL + HR * PR )
                /// H  = Entropy of Current Node.
                /// HL = Entropy of Left child of current node.
                /// HR = Entropy of right child of current node.
                /// PL = Probability of values occurrences in Left node.
                /// PR = Probability of values occurrences in Right node.


                double TotalGain = 0.0;

                /// Adding  --> H
                for( auto& x: MainNodeFreq ) {
                        if( x.second <= 0 ) continue;
                        double T   = ( x.second * 1.0 ) / ( numberOfExamples * 1.0 );
                        TotalGain += -( T ) * log2( T );
                }

                /// Subtracting --> ( HL * PL )
                double PL = double(numberOfExampleInLeft) / double(numberOfExamples);

                for( auto& x: LeftNodeFreq ) {
                        if( x.second <= 0 ) continue;
                        double T  = ( x.second * 1.0 ) / ( numberOfExampleInLeft * 1.0 );
                        double HL = - ( T ) * log2( T );
                        TotalGain -= ( HL * PL );
                }

                /// Subtracting --> ( HR * PR )
                double PR = double(numberOfExampleInRight) / double(numberOfExamples);

                for( auto& x: RightNodeFreq ) {
                        if( x.second <= 0 ) continue;
                        double T  = (x.second * 1.0) / (numberOfExampleInRight*1.0);
                        double HR = - ( T ) * log2( T );
                        TotalGain -= ( HR * PR );
                }

                return TotalGain;
        }

        unordered_map<int,double> Distribution( vector< vector<double> >& examples ) {

                unordered_map<int,int> FreqCount;
                unordered_map<int,double> distribution;

                /// These Stuffs are just used to make unordered_map faster, nothing to do with actual task.
                FreqCount.max_load_factor( 0.25 );      FreqCount.reserve( 1 << 10 );
                distribution.max_load_factor( 0.25 );   distribution.reserve( 1 << 10 );
                /// ---------------------------------------------------------------------------------------

                for( auto& x: examples ) {
                        FreqCount[ int( x.back( ) ) ] ++;
                }

                int totalExample = examples.size( );

                for( auto& x: FreqCount ) {
                        distribution[ x.first ] = ( 1.0 * x.second ) / ( 1.0 * totalExample );
                }
                return distribution;
        }

        tuple<int,double,double> ChooseAttribute(
                                 vector< vector<double> >& examples, vector<int>& attributes, bool isOptimized ) {

                int bestAttribute = -1;
                double bestGain = -1, bestThreshold = -1;

                auto min_max = [&]( int attribute )->tuple<double,double>{
                        double minn = examples[0][attribute], maxx = minn;
                        for( auto& x: examples ) {
                                minn = min( minn, x[ attribute ] );
                                maxx = max( maxx, x[ attribute ] );
                        }
                        return make_tuple(minn, maxx);
                };

                auto GetbestOutcome = [&]( int attribute ) {
                        tuple<double,double> extremes = min_max( attribute );

                        double L = get<MIN>(extremes);
                        double M = get<MAX>(extremes);

                        for( int k = 1; k <= 50; k ++ ) {
                                double threshold = L + k * ( M - L) / 51;
                                double gain      = InformationGain( examples, attribute, threshold );
                                if( gain > bestGain ) {
                                        bestGain = gain;
                                        bestAttribute = attribute;
                                        bestThreshold = threshold;
                                }
                        }
                };

                if( !isOptimized ) {
                        srand( int( time( 0 ) ) );
                        int attribute = rand( ) % int( attributes.size( ) );
                        GetbestOutcome( attribute );
                }
                else {
                        for( auto& x: attributes )
                                GetbestOutcome( x );
                }

                return make_tuple( bestAttribute, bestThreshold, bestGain );
        }

        bool AllSameClass( vector< vector<double> >& examples ) {
                set<int> check;
                for( auto& x : examples ) {
                        auto item = int(x.back());
                        check.insert( item );
                }
                return check.size() == 1;
        }

        void BuildDescisionTree( Node*& Curr, vector< vector<double> >& examples,
                                 vector<int>& attributes, int PruningThreshold, bool isOptimized, bigint nodeId ) {


                if( AllSameClass( examples ) || examples.size( ) < PruningThreshold ) {
                        Curr = new Node( true, Distribution( examples ), nodeId );
                        return;
                }

                tuple<int,double,double> BestChoice = ChooseAttribute( examples, attributes, isOptimized );
                Curr = new Node( get<ATTRIBUTE>(BestChoice), get<THRESHOLD>(BestChoice), get<GAIN>(BestChoice), false, nodeId );

                Curr->Extend( );
                vector< vector<double> > leftChildElem, rightChildElem;

                for( auto& x: examples ) {
                        if( x[ get<ATTRIBUTE>(BestChoice) ] < get<THRESHOLD>(BestChoice) )
                                leftChildElem.push_back( x );
                        else
                                rightChildElem.push_back( x );
                }

                bigint L = nodeId * 2, R = L + 1;

                if( !leftChildElem.empty() )
                        BuildDescisionTree( Curr->Left,  leftChildElem,  attributes, PruningThreshold, isOptimized, L );
                if( !rightChildElem.empty() )
                        BuildDescisionTree( Curr->Right, rightChildElem, attributes, PruningThreshold, isOptimized, R );
        }

        void Train( string dataFile, string option ) {

                vector<int> attrs;
                vector< vector<double> > examples;
                PrepareTrainingData( dataFile, examples, attrs );

                bigint RootID = bigint(1);

                if( option == "optimized" ) {
                        SetForestSize( 1 );
                        Node*& tree = Forest.front( );
                        BuildDescisionTree( tree, examples, attrs, GetPruningThreshold( ), true, RootID );
                }
                else if( option == "randomized" ) {
                        SetForestSize( 1 );
                        Node*& tree = Forest.front( );
                        BuildDescisionTree( tree, examples, attrs, GetPruningThreshold( ), false, RootID );
                }
                else if( option == "forest3" ) {
                        SetForestSize( 3 );
                        for( auto& tree: Forest ) {
                                BuildDescisionTree( tree, examples, attrs, GetPruningThreshold( ), false, RootID );
                        }
                }
                else if( option == "forest15" ){
                        SetForestSize( 15 );
                        for( auto& tree: Forest ) {
                                BuildDescisionTree( tree, examples, attrs, GetPruningThreshold( ), false, RootID );
                        }
                }
        }

        unordered_map<int,double> GetPredictedDistribution( Node*& curr, vector<double>& testObj ) {
                if( curr->isLeaf  )
                        return curr->distribution;

                if( curr->Left != NULL )
                        if( curr->Left->inited && testObj[ curr->attribute ] < curr->threshold )
                                return GetPredictedDistribution( curr->Left, testObj );
                else if( curr->Right != NULL  )
                        if( curr->Right->inited )
                                return GetPredictedDistribution( curr->Right, testObj );
        }

        void ClassifyObject( vector<double>& testObj, int objId,
                             double& classificationAccuracy, int& correctPrediction, ofstream& out ) {

                srand( int( time( 0 ) ) );

                unordered_map<int,double> FinalDistribution;
                vector< unordered_map<int,double> > distributions;

                for( auto& tree: Forest )
                        distributions.push_back( GetPredictedDistribution( tree, testObj ) );

                double numberOfDistributions = 1.0 * distributions.size();

                for( auto& distribution: distributions ) {
                        for( auto& x: distribution ) {
                                FinalDistribution[ x.first ] += ( x.second / numberOfDistributions );
                        }
                }

                double maxProbability = 0.0; int PredictedClass = 0;

                for( auto& x: FinalDistribution ) {
                        if( maxProbability < x.second ) {
                                PredictedClass = x.first;
                                maxProbability = x.second;
                        }
                }

                vector<int> tiedElements;
                for( auto& x: FinalDistribution ) {
                        if( x.second == maxProbability )
                                tiedElements.push_back( x.first );
                }

                if( tiedElements.size() == 1 ) {
                        double accuracy = PredictedClass == int(testObj.back( )) ? 1.0 : 0.0;

                        classificationAccuracy += accuracy;
                        if( accuracy > 0.0 ) correctPrediction ++;

                        PrintTestResult( out, objId, PredictedClass, int(testObj.back()), accuracy );
                }
                else {
                        double accuracy = 0.0;
                        PredictedClass = rand() % int(  tiedElements.size() );

                        if( std::find( tiedElements.begin(), tiedElements.end(), int(testObj.back()) ) != tiedElements.end() )
                                accuracy = 1.0 / double( tiedElements.size() );
                        else    accuracy = 0.0;

                        classificationAccuracy += accuracy;
                        if( accuracy > 0.0 ) correctPrediction ++;

                        PrintTestResult( out, objId, PredictedClass, int(testObj.back()), accuracy );
                }
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

        void PrepareTrainingData( string dataFile, vector< vector<double> >& examples, vector<int>& attrs  ) {
                ifstream inFile( dataFile );

                if( !inFile ) {
                        cout << "Data File Not Found!"  << endl;
                        return;
                }

                PrepareData( inFile, examples );

                for( int i = 0; i < examples[ 0 ].size( ) - 1; i ++ )
                        attrs.push_back( i );
        }

        void PrepareTestData( string dataFile, vector< vector<double> >& TestData ) {

                ifstream inFile( dataFile );

                if( !inFile ) {
                        cout << "Data File Not Found!"  << endl;
                        return;
                }

                PrepareData( inFile, TestData );
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

        void PrintTestResult( ofstream& out, int& id, int& Predicted, int True, double& accuracy  ) {
                /// Outputs in a file.
                out <<"ID="<<setw(5)<<right<<id<<", predicted="<<setw(3)<<right<<Predicted;
                out <<", true="<<setw(3)<<right<<True<<", accuracy="<<setprecision(4)<<accuracy<<endl;

                /// Outputs in console.
                cout <<"ID="<<setw(5)<<right<<id<<", predicted="<<setw(3)<<right<<Predicted;
                cout <<", true="<<setw(3)<<right<<True<<", accuracy="<<setprecision(4)<<accuracy<<endl;
        }

        void PrintFormatted( std::ofstream& out, int treeId, Node*& curr, bool isLeaf ) {
                bigint nodeId; int attribute; double threshold, gain;

                if( isLeaf ) nodeId = curr->nodeId, attribute = -1, threshold = -1.0, gain = 0.0;
                else         nodeId = curr->nodeId, attribute = curr->attribute, threshold = curr->threshold, gain = curr->gain;

                /// Outputs result into a file.
                out << "tree=" << setw(2) << right << treeId << ", node=" << setw(90) << right << nodeId.toString() << ", feature=" << setw(2) << right << attribute;
                out << ", thr=" << setw(6) << right << setprecision(2) << threshold << ", gain=" <<setw(24)<<right<<setprecision(20) << gain << endl;


                /// Outputs result into Console.
                cout << "tree=" << setw(2) << right << treeId << ", node=" << setw(90) << right << nodeId.toString() << ", feature=" << setw(2) << right << attribute;
                cout << ", thr=" << setw(6) << right << setprecision(2) << threshold << ", gain=" <<setw(24)<<right<<setprecision(20) << gain << endl;
        }

        void PrintForest( string outFile ) {
                ofstream out(outFile);
                int treeId = 0;
                for( auto& tree: Forest ) {
                        PrintTree( tree, treeId ++, out );
                }
        }

        void PrintTree( Node*& Tree, int treeId, ofstream& out ) {
                if( Tree == NULL ) {
                        out  << "Tree is Empty !" << endl;
                        cout << "Tree is Empty !" << endl;
                        return;
                }

                queue<Node*> Q; Q.push( Tree );

                while( !Q.empty( ) ) {
                        Node* curr = Q.front( ); Q.pop( );

                        if( !curr->inited ) continue;

                        if( curr->isLeaf ) {
                                PrintFormatted( out, treeId, curr, true );
                                continue;
                        }
                        else    PrintFormatted( out, treeId, curr, false );

                        if( curr->Left != NULL  )
                                if( curr->Left->inited )  Q.push( curr->Left );
                        if( curr->Right != NULL  )
                                if( curr->Right->inited ) Q.push( curr->Right );
                }
        }
};
