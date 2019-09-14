using NeuralNetwork.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Classes
{
    public enum ActivationFunctions
    {
        Sigmoid,
        ReLU
    }

    public class Layer
    {
        public Layer(ActivationFunctions activation, int numberOfNeurons, bool isFullyConnected)
        {

        }

        public ActivationFunctions ActivationFunction { get; set; } = ActivationFunctions.Sigmoid;
    }
}