#include <Eigen/Dense>
#include <string>

class Ann {
    private:
    int                 numberOfLayers;
    int                 numberOfInputs;
    int                 numberOfOutputs;
    Layer*              layers; 
    double              Error;
    std::string         activationFunction;
    std::string         lossFunction;
    Eigen::VectorXd     output;

    // ---------------------------<core private functions>---------------------------
    void parseWeightsAddress();                             // get weights from file
    void parseInputOutputAddress();                         // get inputs from file
    void makeLayers();



    // ---------------------------<activation functions>---------------------------
    void sigmoid();                                         // Logistic
    void tanh();                                            // Hyperbolic tanget
    void relu();                                            // rectified Linear Unit
    void leakyRelu();                                       
    void elu();                                             // exponential linear unit
    void swish();
    void softmax();

    // ---------------------------<loss functions>---------------------------
    void MSE();                                             // Mean Squared Error
    void MAE();                                             // Mean Absolute Error
    void CrossEntropyLoss();                                
    void HuberLoss();
    void BinaryCrossEntropyLoss();
    void CategoricalCrossEntropyLoss();
    void HingeLoss();
    void KLDivergence();                                    // Kullback-Leibler Divergence
    void SquaredHingeLoss();
    void PoissonLoss();

    // ---------------------------<forward propogation>---------------------------
    void forwradPropogtionEvaluate();
    void forwardPropogationTrain();
    void forwardPropogation();

    // ---------------------------<backward propogation>---------------------------
    
    void backwardPropogation();


    public:
    Ann(
        int numLayer,
        int numInput,
        int numOutput,
        std::string& activationFunction,
        std::string& lossFunction
    );                                                      // random manual network building

    Ann(
        std::string& weightAddress,
        std::string& inputAddress,
        std::string& outputAddress,
        std::string& activationFunction,
        std::string& lossFunction
    );                                                      // load a neural network from a file
    
    const int& getNumberOfLayers () const;
    const int& getNumberOfInputs () const;
    const int& getNumberOfOutputs () const;
    const double& getError () const;

    void saveToFile(const std::string& address) const;              // save current ann to file
    void resetWeightsToRandom();                                    // reset weights to random 


    void train(std::string trainType);
    
    
};

class Layer {
    friend class Ann;

    private:
    Eigen::MatrixXd         weights;
    const int               numberOfNodes;                  // weight column
    const int               previousLayerNumberOfNodes;     // weight rows
    
    Layer(const int& numRows, const int& numColumns);       // random valued matrix
    Layer(const Eigen::MatrixXd& m);                        // predefined matrix

    Eigen::MatrixXd& getMatrix ();              // return a const reference to matrix
    const int& getNumberOfNodes () const;
    const int& getPreviousLayerNumberOfNodes () const;

};