import urllib.request
import urllib.error
import csv

num_downloaded=0
with open('facescrub_actors.txt', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            pass
        else:
            print('Saving image ' + str(num_downloaded) + '...')
            # call url and save image
            try:
                url = row[3]
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'}) 
                image_res = urllib.request.urlopen(req, None, 5)
                out = open('./images/' + str(num_downloaded) + '_facescrub.' + str(row[3].split('.')[-1]), 'wb')
                out.write(image_res.read())
                out.close()
                num_downloaded += 1
            except urllib.error.HTTPError as err:
                print('Error for actor image_id ' + row[1] + ' error {0}'.format(err))
            except urllib.error.URLError as err:
                print('Error for actor image_id ' + row[1] + ' error {0}'.format(err))
            except KeyboardInterrupt:
                raise
            except:
                print('Error for actor image_id ' + row[1])

with open('facescrub_actresses.txt', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            pass
        else:
            print('Saving image ' + str(num_downloaded) + '...')
            # call url and save image
            try:
                url = row[3]
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'}) 
                image_res = urllib.request.urlopen(req, None, 5)
                out = open('./images/' + str(num_downloaded) + '_facescrub.' + str(row[3].split('.')[-1]), 'wb')
                out.write(image_res.read())
                out.close()
                num_downloaded += 1
            except urllib.error.HTTPError as err:
                print('Error for actor image_id ' + row[1] + ' error {0}'.format(err))
            except urllib.error.URLError as err:
                print('Error for actor image_id ' + row[1] + ' error {0}'.format(err))
            except KeyboardInterrupt:
                raise
            except:
                print('Error for actor image_id ' + row[1])


