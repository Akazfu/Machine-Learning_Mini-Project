import csv

with open('adult.test') as input_file:
   lines = input_file.readlines()
   newLines = []
   for line in lines:
      newLine = line.strip().split(",")
      if newLine[-1] == " <=50K":
          newLine[-1] = 0;
      elif newLine[-1] == " >50K":
          newLine[-1] = 1;
      #print(newLine[-1])


      newLines.append( newLine )

with open('test.csv', 'wb') as test_file:
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )
