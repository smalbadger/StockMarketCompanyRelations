COI = [
        "NVIDIA CORP",
        "APPLE INC",
        "ADVANCED MICRO DEVICES INC",
        "TAIWAN SEMICONDUCTOR MFG CO LTD",
        "INTEL CORP"
    ]
    
dataDir = "../data/"

filenames = ["crsp16_1.csv","crsp16_2.csv","crsp17_1.csv","crsp17_2.csv"]

for filename in filenames:
    inFile = open(dataDir + filename,"r")
    outFile = open(dataDir + filename + "_trimmed","w")
    
    for line in inFile:
        if 'COMNAM' in line:
            outFile.write(line)
            
        for company in COI:
            if company in line:
                outFile.write(line)
                break
    
    inFile.close()
    outFile.close()
