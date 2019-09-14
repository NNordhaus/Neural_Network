using NeuralNetwork.Classes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Drawing.Imaging;
using System.Drawing;
using System.Security.Cryptography;

namespace NeuralNetwork.NMIST
{
    public class NMIST_Loader
    {
        public static List<DataSet> GetDataSet(string imageFilePath, string labelFilePath, int count)
        {
            var dataSets = new List<DataSet>(count);

            /*
            TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
            [offset] [type]          [value]          [description] 
            0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
            0004     32 bit integer  60000            number of items 
            0008     unsigned byte   ??               label 
            0009     unsigned byte   ??               label 
            ........ 
            xxxx     unsigned byte   ??               label
            The labels values are 0 to 9.
            */
            using (var fileStream = File.OpenRead(labelFilePath))
            {
                // skip the first 8 bytes
                fileStream.Seek(8, SeekOrigin.Begin);

                for (int i = 0; i < count; i++)
                {
                    var label = fileStream.ReadByte();
                    var targets = new double[10];
                    targets[label] = 1.0;
                    dataSets.Add(new DataSet(null, targets));
                }
            }

            /*
            IMAGE FILE (train-images-idx3-ubyte):
            [offset] [type]          [value]          [description] 
            0000     32 bit integer  0x00000803(2051) magic number 
            0004     32 bit integer  60000            number of images 
            0008     32 bit integer  28               number of rows 
            0012     32 bit integer  28               number of columns 
            0016     unsigned byte   ??               pixel 
            0017     unsigned byte   ??               pixel 
            ........ 
            xxxx     unsigned byte   ??               pixel
            */

            using (var fileStream = File.OpenRead(imageFilePath))
            {
                // skip the first 16 bytes
                fileStream.Seek(16, SeekOrigin.Begin);

                int imageSize = 28 * 28;

                for (int i = 0; i < count; i++)
                {
                    var imagePixels = new byte[imageSize];
                    fileStream.Read(imagePixels, 0, imageSize); 

                    dataSets[i].Values = imagePixels.Select(p => (double)p / 255d).ToArray();
                }
            }

            return dataSets;
        }

        public static void SaveImageTosDisk(List<DataSet> dataSet, string folderPath)
        {
            Parallel.ForEach(dataSet, (i) =>
            {
                var bmp = new Bitmap(28, 28, PixelFormat.Format24bppRgb);
                for(int x = 0; x < 28;  x++)
                {
                    for (int y = 0; y < 28; y++)
                    {
                        int darkness = (int)(i.Values[(y * 28) + x] * 255);
                        bmp.SetPixel(x, y, Color.FromArgb(255, 255-darkness, 255-darkness, 255-darkness));
                    }
                }

                // Figure out the label 
                var label = i.Targets.ToList().IndexOf(i.Targets.Max()).ToString() + "_";

                // Figure out the hash
                using (SHA256 crypto = SHA256.Create())
                {
                    List<byte> bytes = new List<byte>(28*28*8);

                    for (int j = 0; j < (28 * 28); j++)
                    {
                        bytes.AddRange(BitConverter.GetBytes(i.Values[j]));
                    }

                    label += BitConverter.ToString(crypto.ComputeHash(bytes.ToArray())).Replace("-", string.Empty).Substring(0, 32);
                }

                bmp.Save(folderPath + label + ".png", ImageFormat.Png);
            });
        }
    }
}