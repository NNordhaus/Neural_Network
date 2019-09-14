using NeuralNetwork.Classes;
using NeuralNetwork.NMIST;
using NeuralNetwork.Tools;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //var testSet = NMIST_Loader.GetDataSet("test-images.data", "test-labels.data", 10000);
            //NMIST_Loader.SaveImageTosDisk(testSet, @"C:\Users\Neil\Desktop\My Repos\NN_1\TestImages\");
            //return;

            Console.WriteLine("Loading Training Data...");
            
            var trainingSet = NMIST_Loader.GetDataSet("train-images.data", "train-labels.data", 60000);

            var nn = new Network(784, new int[] { 422, 280 }, 10, 0.3, 0.49);

            Console.WriteLine(nn.Config);

            var testSet = NMIST_Loader.GetDataSet("test-images.data", "test-labels.data", 10000);

            Console.WriteLine("Starting Failure Rate: " + nn.Fitness_FailureRate(testSet).ToString("0.000000"));

            nn.Train(trainingSet, testSet, 40);
            

            Console.WriteLine(nn.Config);
            Console.Beep(555, 620);

           
            Console.ReadLine();

            Console.ReadLine();
        }
    }
}