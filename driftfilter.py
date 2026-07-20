import json, math, random

def near(x, centers):
    return min(range(len(centers)), key=lambda k: sum((a-b)**2 for a,b in zip(x,centers[k])))

def run(seed=19):
    rng=random.Random(seed)
    frozen=[[-2.,0.],[2.,0.]]; adaptive=[c[:] for c in frozen]
    f_ok=a_ok=n=0
    for t in range(240):
        shift=3.4*t/239
        label=t%2
        true=[(-2 if label==0 else 2)+shift, .7*math.sin(t/35)]
        x=[true[0]+rng.gauss(0,.45),true[1]+rng.gauss(0,.45)]
        pf=near(x,frozen); pa=near(x,adaptive)
        f_ok+=pf==label; a_ok+=pa==label; n+=1
        rate=.08
        adaptive[pa]=[(1-rate)*v+rate*z for v,z in zip(adaptive[pa],x)]
    fa,aa=f_ok/n,a_ok/n
    return {"frozen_accuracy":round(fa,3),"filtered_accuracy":round(aa,3),
            "accuracy_gain_pct":round(100*(aa-fa),1)}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
