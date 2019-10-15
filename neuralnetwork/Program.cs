using NeuralNetwork.Classes;
using NeuralNetwork.Tools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
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
            //NMIST_Loader.SaveImageTosDisk(testSet, @"C:\Other_Repos\Test Images\");
            //return;

            //var nn = new Network(3, new int[] { 2, 3 }, 3, 0.4, 0.2);
            //File.WriteAllText("output.json", nn.Serialize(true));

            var dsl = new DataSets.NMIST_Loader();

            Console.WriteLine("Loading Training Data...");
            var trainingSet = dsl.GetDataSet("train-images.data", "train-labels.data", 60000);

            Console.WriteLine("Loading Test Data...");
            var testSet = dsl.GetDataSet("test-images.data", "test-labels.data", 10000);

            // Use this to load a saved network from JSON
            //var nn = Network.FromJson(File.ReadAllText("Network 2019-10-02.json"));

            // Use this to create a new network from scratch
            var nn = new Network(784, new int[] { 400, 280 }, 10, 0.28, 0.4);
            
            Console.WriteLine(nn.Config);

            //nn.Fitness_FailureRate(testSet);
            
            nn.Train(trainingSet, testSet, 40);
            
            Console.WriteLine(nn.Config);
            Console.Beep(555, 620);

            Console.WriteLine("Save this Network? Y/N");

            var save = Console.ReadLine();
            if (save.ToUpperInvariant() == "Y")
            {
                var fileName = DateTime.Now.ToString("yyyy-MM-dd_HHmmss") + ".json";
                Console.WriteLine("Saving " + fileName + "...");
                File.WriteAllText("Network " + fileName, nn.Serialize());
                Console.WriteLine("Saved.");
            }

            Console.WriteLine("Press any key to exit");

            Console.ReadKey();
        }
    }
}