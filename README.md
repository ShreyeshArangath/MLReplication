# MLReplication

## Set up

Download the RockYouDataset from [here](https://drive.google.com/file/d/1SOSNYS2db09XrU7bX-VJsv8est1qD2Fu/view?usp=sharing) and move the file to ./Data/

## Results

Isolating the feature set to just a single dimensional entity (inter-key) brings down the overall prediction ability of the classifier. The accuracy score drops from ~16.5% to ~5.06%. 

MSU, GREYC Web, GREYC 
1. Accuracy Score: 15.68%. 
2. Number of unique digraphs: 307 

GREYC Web, GREYC 
1. Accuracy Score: 19.80%
2. Number of unique digraphs: 255 

## Initial experiment results: 

thresholdValue = 200

### Top 10 Passwords

1. password — Penalty:594 Guess: 3  Occurences: 59462
2. princess — Penalty:334 Guess: 5  Occurences: 33291
3. michelle — Penalty:527 Guess: 23  Occurences: 12714
4. sunshine — Penalty:940 Guess: 25  Occurences: 11489
5. samantha — Penalty:631 Guess: 64  Occurences: 7717
6. danielle — Penalty:423 Guess: 77  Occurences: 7148
7. jonathan — Penalty:623 Guess: 80  Occurences: 6775
8. princesa — Penalty:691 Guess: 99  Occurences: 6027
9. estrella — Penalty:445 Guess: 103  Occurences: 5850
10. carolina — Penalty:401 Guess: 128  Occurences: 5126

 Theshold EXPERIMENT 

1. password — Penalty:1513 Guess: 3  Occurences: 59462
2. princess — Penalty:1026 Guess: 5  Occurences: 33291
3. michelle — Penalty:907 Guess: 23  Occurences: 12714
4. sunshine — Penalty:1425 Guess: 25  Occurences: 11489
5. samantha — Penalty:1079 Guess: 64  Occurences: 7717
6. danielle — Penalty:846 Guess: 77  Occurences: 7148
7. jonathan — Penalty:1010 Guess: 80  Occurences: 6775
8. princesa — Penalty:1017 Guess: 99  Occurences: 6027
9. estrella — Penalty:1270 Guess: 103  Occurences: 5850
10. carolina — Penalty:1027 Guess: 128  Occurences: 5126


### Bottom 10 Passwords

1. 0melting — Penalty:327 Guess: 13482300  Occurences: 1
2. 0melette — Penalty:219 Guess: 13482305  Occurences: 1
3. 0melanie — Penalty:251 Guess: 13482306  Occurences: 1
4. 0megaman — Penalty:327 Guess: 13482310  Occurences: 1
5. 0masadao — Penalty:335 Guess: 13482344  Occurences: 1
6. 0marcito — Penalty:340 Guess: 13482366  Occurences: 1
7. 0mandrin — Penalty:61 Guess: 13482385  Occurences: 1
8. 0madness — Penalty:211 Guess: 13482414  Occurences: 1
9. 0madison — Penalty:437 Guess: 13482415  Occurences: 1
10. 0machine — Penalty:281 Guess: 13482417  Occurences: 1


 Theshold EXPERIMENT 


1. 0melting — Penalty:1015 Guess: 13482300  Occurences: 1
2. 0melette — Penalty:1066 Guess: 13482305  Occurences: 1
3. 0melanie — Penalty:788 Guess: 13482306  Occurences: 1
4. 0megaman — Penalty:632 Guess: 13482310  Occurences: 1
5. 0masadao — Penalty:637 Guess: 13482344  Occurences: 1
6. 0marcito — Penalty:945 Guess: 13482366  Occurences: 1
7. 0mandrin — Penalty:845 Guess: 13482385  Occurences: 1
8. 0madness — Penalty:750 Guess: 13482414  Occurences: 1
9. 0madison — Penalty:763 Guess: 13482415  Occurences: 1
10. 0machine — Penalty:737 Guess: 13482417  Occurences: 1