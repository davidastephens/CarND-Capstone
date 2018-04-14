import csv
import os

def delete(f):
    try:
        os.remove(f)
    except:
        pass

d = {
        'trafficLightGreen': 3,
        'trafficLight': 4, #Unknown
        'trafficLightRed': 1,
        'trafficLightYellow': 2,
        'trafficLightYellowLeft': 2,
        'trafficLightRedLeft': 1,
        'trafficLightGreenLeft': 1,
    }

keep_files = []
all_files = []

with open('labels.csv', 'r') as csvfile:
    label_reader = csv.reader(csvfile, delimiter=' ')
    with open('filtered_labels.csv', 'w') as outfile:

        column_names = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        label_writer = csv.writer(outfile, delimiter=',')
        label_writer.writerow(column_names)

        for row in label_reader:
            filename = row[0]
            xmin = int(row[1])
            ymin = int(row[2])
            xmax = int(row[3])
            ymax = int(row[4])
            occluded = row[5]
            label = row[6]
            if len(row) == 8:
                attribute = row[7]
            else:
                attribute = ''
            cls = label + attribute
            all_files.append(filename)

            if 'trafficLight' in row and occluded == '0':
                cls = d[cls]
                newrow = [filename, cls, xmin, ymin, xmax, ymax]
                label_writer.writerow(newrow)
                keep_files.append(filename)


for f in all_files:
    if f not in keep_files:
        delete(f)

