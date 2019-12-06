import csv


with open('dataset/income.data') as input_file:
   lines = input_file.readlines()
   newLines = []

   for line in lines:
      newLine = line.strip().split(",")
      if newLine[-1] == " <=50K":
          newLine[-1] = 0
      elif newLine[-1] == " >50K":
          newLine[-1] = 1
      for i in range(len(newLine)):
          if newLine[i] ==" ?":
              newLine[i]=""

      #print(newLine[-1])
      newLines.append( newLine )


with open('dataset/test.csv', 'w') as test_file:
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )
