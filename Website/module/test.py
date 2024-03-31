import os
# assign directory
directory = '/Users/shankar-tt0027/Documents/Eclipse/deployment/onezoho/AdventNet/Sas/tomcat/bin/'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print('<classpathentry kind="lib" path="'+f+'"/>')