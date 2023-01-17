import os

# os.system("awk -F, '{print \"2023-01-05,12-51-28\" f1,$1,$2,$3}' OFS=,\
#  ../../outputs/engine.model.vgg.VGG11/2023-01-05/12-51-28/metrics/epoch_metrics.csv  > log_val.ID.csv")
# os.system("tail -n +2 log_val.ID.csv >> ../log_val.SO.csv")
# os.system("head -n 1 log_val.ID.csv > head.csv")
# os.system("awk -F, '{print \"Date,time\" f1,$3,$4,$5}' OFS=, head.csv  > head.ID.csv")
# os.system("cat head.ID.csv ../log_val.SO.csv > ../log_val.S.csv")
# os.system("rm ../log_val.SO.csv")

dir_path = '../../outputs'

for dirpath, dirnames, filenames in os.walk(dir_path):
    for filename in filenames:
        if filename.endswith('.csv'):
            if filename == 'epoch_metrics.csv':
                experiment = dirpath.split('/')
                hour = experiment[-2]
                date = experiment[-3]
                model = experiment[-4].split('.')[-1]

                summary_file = f'{model}_{filename[:-4]}_summary.csv'

                print('---EXPERIMENT---')
                print('path', dirpath)
                print('model:', model)
                print('date:', date)
                print('hour:', hour)
                os.system("awk -F, '{print \"%s,%s,%s\" f1,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18}' OFS=,\
                 %s/%s.csv  > %s.ID.csv" % (model, date, hour, dirpath, filename[:-4], filename[:-4]))
                os.system("tail -n +2 %s.ID.csv >> ../%s.SO.csv" % (filename[:-4], filename[:-4]))
                os.system("head -n 1 %s.ID.csv > %s.head.csv" % (filename[:-4], filename[:-4]))
                os.system("awk -F, '{print \"model,date,time\" f1,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21}' OFS=, %s.head.csv  > %s.head.ID.csv" % (filename[:-4], filename[:-4]))




if os.path.exists('../epoch_metrics.S.csv'):
    os.system('rm ../epoch_metrics.S.csv')

os.system("cat epoch_metrics.head.ID.csv ../epoch_metrics.SO.csv > ../epoch_metrics.S.csv")
os.system("rm ../epoch_metrics.SO.csv")
