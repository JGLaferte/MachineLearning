using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworkLib;

namespace NeuralNetwoekConsoleTester
{
    class Program
    {
        static void Main(string[] args)
        {
            MatrixBuilder<double> matrixEngine = Matrix<double>.Build;
            VectorBuilder<double> vectorEngine = Vector<double>.Build;

            Matrix<double> inputs = matrixEngine.Dense(1, 2, new double[] { 1, 1 });
            
            Matrix<double> outputs = matrixEngine.Dense(1, 1, new double[] { 0 });
            NeuralNetwork neuralNetwork = new NeuralNetwork(inputs, 3, outputs);
            neuralNetwork.FowardPropagation();

            Console.WriteLine(neuralNetwork.InputLayer.Inputs);
            Console.WriteLine(neuralNetwork.InputLayer.Weights);

            Console.WriteLine(neuralNetwork.HiddenLayer.Values.ToString());
            Console.WriteLine(neuralNetwork.HiddenLayer.ActivateValues.ToString());
            Console.WriteLine(neuralNetwork.HiddenLayer.Weights.ToString());

            Console.WriteLine(neuralNetwork.OutputLayer.Values.ToString());
            Console.WriteLine(neuralNetwork.OutputLayer.ActivateValues.ToString());
            Console.WriteLine(neuralNetwork.OutputLayer.ExpectedInput.ToString());


            Console.ReadLine();
        }
    }
}
