import matplotlib.pyplot as plt
import numpy as np

def plot_rq2(scores,bcs,fnames,planner):
    N = 4
    plt.rcParams.update({'font.size':15})
    fig, ax = plt.subplots(figsize=(8,8))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.08         # the width of the bars
    result =[]
    for m in range(0,int(len(scores))):
        p25,p50,p75,p100 = 0,0,0,0
        r25,r50,r75,r100 = 0,0,0,0
        a25,a50,a75,a100 = 0,0,0,0
        score = scores[m]
        bugchange = bcs[m]
        for i in range(0,int(len(score))):
            if 0<=score[i]<0.25:
                p25+=1
                if bugchange[i]<0:
                    r25-=bugchange[i]
                if bugchange[i]>0:
                    a25+=bugchange[i]
            if 0.25<=score[i]<0.5:
                p50+=1
                if bugchange[i]<0:
                    r50-=bugchange[i]
                if bugchange[i]>0:
                    a50+=bugchange[i]
            if 0.5<=score[i]<0.75:
                p75+=1
                if bugchange[i]<0:
                    r75-=bugchange[i]
                if bugchange[i]>0:
                    a75+=bugchange[i]
            if 0.75<=score[i]<=1:
                p100+=1
                if bugchange[i]<0:
                    r100-=bugchange[i]
                if bugchange[i]>0:
                    a100+=bugchange[i]
        s = p25+p50+p75+p100
        result.append([p25/s,p50/s,p75/s,p100/s]) if s!=0 else result.append([p25,p50,p75,p100])

    ax.set_ylabel("Ratio of plans over all plans")
    # ax.set_ylabel("Total amount of bugs reduced")
    # ax.set_ylabel("Number of bugs added")
    # ax.set_ylabel("Number of bugs reduced")
    ax.set_xlabel("Overlap percentage")

    p0 = ax.bar(ind-width*4, result[0], width, bottom=0,label=fnames[0][0].split('-')[0])
    p1 = ax.bar(ind-width*3, result[1], width, bottom=0,label=fnames[1][0].split('-')[0])
    p2 = ax.bar(ind-width*2, result[2], width, bottom=0,label=fnames[2][0].split('-')[0])
    p3 = ax.bar(ind-width*1, result[3], width, bottom=0,label=fnames[3][0].split('-')[0])
    p4 = ax.bar(ind+width*0, result[4], width, bottom=0,label=fnames[4][0].split('-')[0])
    p5 = ax.bar(ind+width*1, result[5], width, bottom=0,label=fnames[5][0].split('-')[0])
    p6 = ax.bar(ind+width*2, result[6], width, bottom=0,label=fnames[6][0].split('-')[0])
    p7 = ax.bar(ind+width*3, result[7], width, bottom=0,label=fnames[7][0].split('-')[0])

    ax.set_title(planner)
    ax.set_xticks(ind)
    ax.set_xticklabels(('0-25', '25-50', '50-75', '75-100'))
    ax.autoscale_view()
    plt.grid(axis='y')
    plt.savefig("rq2"+planner,dpi=100,bbox_inches = 'tight')
    return result

def plot_rq3(scores,bcs,fnames,planner):
    N = 4
    plt.rcParams.update({'font.size':15})
    fig, ax = plt.subplots(figsize=(8,8))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.08         # the width of the bars
    result =[]
    for m in range(0,int(len(scores))):
        p25,p50,p75,p100 = 0,0,0,0
        r25,r50,r75,r100 = 0,0,0,0
        a25,a50,a75,a100 = 0,0,0,0
        score = scores[m]
        bugchange = bcs[m]
        for i in range(0,int(len(score))):
            if 0<=score[i]<0.25:
                p25+=1
                if bugchange[i]<0:
                    r25-=bugchange[i]
                if bugchange[i]>0:
                    a25+=bugchange[i]
            if 0.25<=score[i]<0.5:
                p50+=1
                if bugchange[i]<0:
                    r50-=bugchange[i]
                if bugchange[i]>0:
                    a50+=bugchange[i]
            if 0.5<=score[i]<0.75:
                p75+=1
                if bugchange[i]<0:
                    r75-=bugchange[i]
                if bugchange[i]>0:
                    a75+=bugchange[i]
            if 0.75<=score[i]<=1:
                p100+=1
                if bugchange[i]<0:
                    r100-=bugchange[i]
                if bugchange[i]>0:
                    a100+=bugchange[i]
        s = p25+p50+p75+p100
        rate = [a25,a50,a75,a100]
        rate = [r25 - a25, r50 - a50, r75 - a75, r100 - a100]
        result.append(rate)
        #     result.append([p25,p50,p75,p100])
        #     result.append([p25/s,p50/s,p75/s,p100/s]) if s!=0 else result.append([p25,p50,p75,p100])
        # result.append([p25/s,p50/s,p75/s,p100/s]) if s!=0 else result.append([p25,p50,p75,p100])

    ax.set_ylabel("Total amount of bugs reduced")
    ax.set_xlabel("Overlap percentage")

    p0 = ax.bar(ind-width*4, result[0], width, bottom=0,label=fnames[0][0].split('-')[0])
    p1 = ax.bar(ind-width*3, result[1], width, bottom=0,label=fnames[1][0].split('-')[0])
    p2 = ax.bar(ind-width*2, result[2], width, bottom=0,label=fnames[2][0].split('-')[0])
    p3 = ax.bar(ind-width*1, result[3], width, bottom=0,label=fnames[3][0].split('-')[0])
    p4 = ax.bar(ind+width*0, result[4], width, bottom=0,label=fnames[4][0].split('-')[0])
    p5 = ax.bar(ind+width*1, result[5], width, bottom=0,label=fnames[5][0].split('-')[0])
    p6 = ax.bar(ind+width*2, result[6], width, bottom=0,label=fnames[6][0].split('-')[0])
    p7 = ax.bar(ind+width*3, result[7], width, bottom=0,label=fnames[7][0].split('-')[0])

    ax.set_title(planner)
    ax.set_xticks(ind)
    ax.set_xticklabels(('0-25', '25-50', '50-75', '75-100'))
    ax.autoscale_view()
    plt.grid(axis='y')
    plt.savefig("rq3"+planner,dpi=100,bbox_inches = 'tight')
    return result