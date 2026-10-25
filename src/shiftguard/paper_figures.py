"""Publication-oriented, annotated 600-DPI PNG exports from actual saved results.

No PDF/SVG export paths are used. One-seed plots do not invent error bars.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import confusion_matrix, roc_curve

from .metrics import calibration_bins, risk_coverage

NAMES = ['MSP', 'Entropy', 'Distance', 'MSP+Entropy', 'Combined']
LABEL = {'MSP':'MSP', 'Entropy':'Entropy', 'Distance':'Feature distance',
         'MSP+Entropy':'MSP + entropy', 'Combined':'All three (ShiftGuard)'}
COLORS = dict(zip(NAMES, ['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']))
STYLES = dict(zip(NAMES, ['-','--','-.',':','-']))
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def generate(run_dir):
    root = Path(run_dir)
    cfg = json.loads((root/'config.json').read_text())
    dpi = int(cfg.get('figure_dpi',600))
    if dpi < 300:
        raise ValueError('Paper figures require at least 300 DPI')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':10,
        'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
        'grid.color':'#c7cbd0','grid.linewidth':.6,'grid.alpha':.45,
        'lines.linewidth':1.7,'savefig.facecolor':'white'})
    outputs = []
    for metric_path in sorted((root/'metrics').glob('*.json')):
        metrics = json.loads(metric_path.read_text())
        if 'classifier' not in metrics or 'methods' not in metrics:
            continue
        experiment = metrics['experiment']
        with np.load(root/'predictions'/f'{experiment}_scores.npz') as z:
            data = {k:z[k] for k in z.files}
        names = [n for n in NAMES if n in metrics['methods']]
        folder = root/'figures'/experiment
        folder.mkdir(parents=True,exist_ok=True)
        smoke = metrics.get('scope') == 'smoke'
        architecture = 'ResNet-18' if cfg.get('architecture')=='resnet18' else 'Small CNN'
        scope = f'{architecture} · {experiment.replace("_"," ")} · N={len(data["errors"]):,} · seed {cfg["seed"]}'
        if smoke:
            scope = 'SMOKE TEST — ' + scope
        shifted = experiment != 'clean'
        context = (' Synthetic corruption; clean IID guarantee is not asserted.' if shifted else
                   ' One seed; no across-seed confidence intervals are shown.')

        def canvas(title, size=(7.2,4.5), **kwargs):
            fig, ax = plt.subplots(figsize=size, layout='constrained', **kwargs)
            fig.suptitle(title+'\n'+scope, fontsize=11, linespacing=1.5)
            return fig, ax

        def save(fig, name, caption):
            path = folder/f'{name}.png'
            fig.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=.12,
                        metadata={'Title':name,'Description':scope+'. '+caption+context,'Software':'ShiftGuard'})
            plt.close(fig)
            (folder/f'{name}.txt').write_text(scope+'. '+caption+context+'\n')
            outputs.append(str(path))

        def legend(ax):
            ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,handlelength=2.5)

        def heatmap(ax, matrix, rows, columns, label, fmt, vmin=0, vmax=None, cmap='Blues'):
            vmax = float(np.max(matrix)) if vmax is None else vmax
            vmax = max(vmax,1e-6)
            image = ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
            ax.set_xticks(np.arange(len(columns)),columns)
            ax.set_yticks(np.arange(len(rows)),rows)
            ax.tick_params(length=0,pad=7)
            ax.set_xticks(np.arange(-.5,len(columns),1),minor=True)
            ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1.4,alpha=1)
            ax.tick_params(which='minor',bottom=False,left=False)
            for r in range(len(rows)):
                for c in range(len(columns)):
                    value=matrix[r,c]
                    rgba=image.cmap(image.norm(value))
                    luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(c,r,fmt(r,c,value),ha='center',va='center',fontsize=9,
                            color='white' if luminance < .5 else '#15191e')
            ax.figure.colorbar(image,ax=ax,pad=.025,label=label,shrink=.9)

        fig,ax=canvas('Error-detection ROC curves')
        if len(np.unique(data['errors']))==2:
            for name in names:
                fpr,tpr,_=roc_curve(data['errors'],data[name])
                auc=metrics['methods'][name]['auroc']
                ax.plot(fpr,tpr,color=COLORS[name],linestyle=STYLES[name],label=f'{LABEL[name]} ({auc:.3f})')
        else:
            ax.text(.5,.5,'AUROC undefined: only one error class observed',ha='center',transform=ax.transAxes)
        ax.plot([0,1],[0,1],color='#888888',linestyle=':',linewidth=1,label='Chance')
        ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02))
        ax.grid(True);legend(ax)
        save(fig,'roc','Positive class: incorrect prediction. Legend includes error AUROC; higher is better.')

        fig,ax=canvas('Selective prediction: risk–coverage curves')
        largest_risk=0.0
        for name in names:
            coverage,risk=risk_coverage(data['errors'],data[name])
            largest_risk=max(largest_risk,float(risk.max()))
            ax.plot(coverage,risk,color=COLORS[name],linestyle=STYLES[name],label=LABEL[name])
        ax.set(xlabel='Coverage: accepted predictions (%)',ylabel='Risk: accepted error rate (%)',
               xlim=(0,1),ylim=(0,min(1,max(.1,largest_risk*1.12))))
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(True);legend(ax)
        save(fig,'risk_coverage','Lower risk at equal coverage is better. AURC is mean prefix risk; exact score ties use expected random-order prefix risk. Axes include all prefix risks; very small accepted counts have unstable empirical risk.')

        operating=np.array([[metrics['methods'][name][f'risk_at_{k}']*100 for k in [50,70,80,90]] for name in names])
        fig,ax=canvas('Error rates at selected coverage levels',size=(7.2,4.2))
        heatmap(ax,operating,[LABEL[n] for n in names],['50%','70%','80%','90%'],
                'Accepted error rate (%)',lambda r,c,v:f'{v:.2f}%')
        ax.set(xlabel='Requested coverage',ylabel='Rejection method')
        save(fig,'risk_at_coverage_heatmap','Annotated values are observed test error percentages at ceil(coverage × N) accepted predictions. Same examples for every method; lower is better.')

        fig,axes=canvas('Probability calibration',size=(9.5,3.9),nrows=1,ncols=3,sharex=True,sharey=True)
        cases=[(data['probabilities'].max(1),1-data['errors'],'Classifier confidence',metrics['classifier']['ece'],'#0072B2'),
               (data['MSP+Entropy'],data['errors'],'Error: MSP + entropy',metrics['error_probability']['MSP+Entropy']['ece'],'#CC79A7'),
               (data['Combined'],data['errors'],'Error: all three',metrics['error_probability']['Combined']['ece'],'#D55E00')]
        bin_records=[]
        for ax,(p,y,title,ece,color) in zip(axes,cases):
            bins=calibration_bins(p,y)
            x=[b['mean_probability'] for b in bins];observed=[b['outcome_rate'] for b in bins]
            size=[12+65*b['count']/max(r['count'] for r in bins) for b in bins]
            ax.plot([0,1],[0,1],color='#777777',linestyle=':',linewidth=1)
            ax.plot(x,observed,color=color,linewidth=1.3)
            ax.scatter(x,observed,s=size,color=color,edgecolor='white',linewidth=.5,zorder=3)
            ax.set(title=f'{title}\nECE = {ece:.3f}',xlabel='Mean predicted probability',xlim=(0,1),ylim=(0,1))
            ax.set_xticks([0,.25,.5,.75,1]);ax.grid(True)
            bin_records.extend([dict(series=title,**b) for b in bins])
        axes[0].set_ylabel('Observed outcome frequency')
        (folder/'calibration_bins.json').write_text(json.dumps(bin_records,indent=2))
        save(fig,'calibration','15 equal-width bins on [0,1], last includes 1; empty bins omitted. Marker area increases with bin sample count (counts in calibration_bins.json). Sparse bins can fluctuate. Classifier outcome is correctness; detector outcomes are errors.')

        fig,ax=canvas('Classification confusion matrix',size=(8.2,6.1))
        matrix=confusion_matrix(data['labels'],data['predictions'],labels=np.arange(10))
        heatmap(ax,matrix,CLASSES,CLASSES,'Test images (count)',lambda r,c,v:f'{v:.0f}')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right',rotation_mode='anchor')
        ax.set(xlabel='Predicted class',ylabel='True class')
        save(fig,'confusion_matrix','Every cell is annotated with its image count. Rows are true labels; columns are predictions. Uses all saved test examples.')

        fig,ax=canvas('Error-detection and selective-prediction results',size=(10,3.4))
        ax.axis('off')
        columns=['Method','AUROC ↑','AURC ↓','Risk @50%\n(%)','Risk @70%\n(%)','Risk @80%\n(%)','Risk @90%\n(%)']
        rows=[]
        for name in names:
            m=metrics['methods'][name]
            rows.append([LABEL[name],f'{m["auroc"]:.4f}' if m['auroc'] is not None else 'N/A',
                         f'{m["aurc"]:.4f}']+[f'{m[f"risk_at_{k}"]*100:.2f}' for k in [50,70,80,90]])
        table=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center',
                       colWidths=[.28,.12,.12,.12,.12,.12,.12])
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.9)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#d7dce1');cell.set_linewidth(.6)
            if r==0:
                cell.set_facecolor('#e7eef5');cell.set_text_props(weight='bold');cell.set_height(cell.get_height()*1.4)
            elif r%2==0:
                cell.set_facecolor('#f6f8fa')
        save(fig,'results_table','AUROC higher is better; AURC and accepted risks lower are better. Risks are percentages; AUROC/AURC are fractions. N/A means undefined. No across-seed uncertainty is implied.')

        risk_rows=metrics.get('risk_rows',[])
        if risk_rows:
            budgets=sorted(set(r['calibration_n'] for r in risk_rows));targets=sorted(set(r['target_risk'] for r in risk_rows))
            largest=max(budgets)
            selected={(r['method'],r['target_risk']):r for r in risk_rows if r['calibration_n']==largest}
            matrix=np.array([[selected[n,t]['test_coverage']*100 for t in targets] for n in names])
            fig,ax=canvas('Frozen clean-rule coverage' if shifted else 'Coverage at statistically controlled risk',size=(7.2,4.6))
            def annotation(r,c,v):
                entry=selected[names[r],targets[c]]
                return f'{v:.1f}%' if entry['status']=='certified' else '0.0%\nNot certified'
            heatmap(ax,matrix,[LABEL[n] for n in names],[f'Target risk {t:.0%}' for t in targets],
                    'Accepted test predictions (%)',annotation,vmin=0,vmax=100,cmap='YlGnBu')
            ax.set(xlabel=f'Independent calibration budget: {largest:,} examples',ylabel='Rejection method')
            save(fig,'risk_control_coverage_heatmap','Thresholds were fixed using independent clean calibration data and simultaneous one-sided binomial upper bounds. Not certified means no eligible rule; reject all and leave conditional risk undefined. Reported coverage is measured on test data. '+('Clean guarantees do not transfer to corruption.' if shifted else 'Confidence applies to the prespecified finite family under IID assumptions, not to each image.'))

            fig,axes=canvas('Coverage versus calibration-label budget',size=(9,4.3),nrows=1,ncols=len(targets),sharey=True,squeeze=False)
            for ax,t in zip(axes[0],targets):
                for name in names:
                    values=[next(r['test_coverage']*100 for r in risk_rows if r['method']==name and r['target_risk']==t and r['calibration_n']==b) for b in budgets]
                    ax.plot(budgets,values,'o',linestyle=STYLES[name],color=COLORS[name],markersize=4,label=LABEL[name])
                ax.set(title=f'Target risk ≤ {t:.0%}',xlabel='Independent calibration examples',ylim=(-2,102))
                ax.set_xticks(budgets);ax.yaxis.set_major_formatter(PercentFormatter(100));ax.grid(True)
            axes[0,0].set_ylabel('Accepted test predictions (%)');legend(axes[0,-1])
            save(fig,'calibration_budget_coverage','Candidate grids are fixed using fitting data. A single Bonferroni family covers every displayed method, threshold, and budget. Zero coverage may indicate no certificate; see the annotated heatmap/CSV. Lines connect separate prespecified budget evaluations; no across-seed error bars.')

            fig,ax=canvas('Risk-control audit at the largest calibration budget',size=(10.5,4.9))
            ax.axis('off')
            columns=['Method','Target\nrisk','Cal.\naccepted','Cal. upper\nrisk','Test\ncoverage','Test\nrisk','Status']
            rows=[]
            for target in targets:
                for name in names:
                    r=selected[name,target]
                    rows.append([LABEL[name],f'{target:.0%}',str(r['calibration_accepted']),
                        f'{r["calibration_upper_risk"]:.2%}' if r['calibration_upper_risk'] is not None else '—',
                        f'{r["test_coverage"]:.2%}',f'{r["test_risk"]:.2%}' if r['test_risk'] is not None else 'N/A',
                        'Certified' if r['status']=='certified' else 'Not certified'])
            table=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center',colWidths=[.26,.10,.11,.13,.12,.12,.16])
            table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.65)
            for (r,c),cell in table.get_celld().items():
                cell.set_edgecolor('#d7dce1');cell.set_linewidth(.6)
                if r==0:
                    cell.set_facecolor('#e7eef5');cell.set_text_props(weight='bold');cell.set_height(cell.get_height()*1.5)
                elif r%2==0:
                    cell.set_facecolor('#f6f8fa')
            save(fig,'risk_control_table','Cal. upper risk is the simultaneous upper bound used for eligibility; test risk is an observed fraction and can fluctuate above target. N/A denotes no accepted test examples. A clean calibration certificate is not a shift guarantee.')

    training_path=root/'training.json'
    if training_path.exists():
        t=json.loads(training_path.read_text());history=t['history']
        if history:
            folder=root/'figures'/'training';folder.mkdir(parents=True,exist_ok=True)
            epochs=[r['epoch'] for r in history]
            fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
            fig.suptitle(f'Training diagnostics · seed {cfg["seed"]}'+(' · SMOKE TEST' if cfg['scope']=='smoke' else ''),fontsize=11)
            cases=[('train_loss','Training cross-entropy','#0072B2'),('selection_accuracy','Selection accuracy (%)','#009E73'),('learning_rate','Learning rate','#D55E00')]
            for ax,(key,label,color) in zip(axes,cases):
                values=[r[key]*(100 if key=='selection_accuracy' else 1) for r in history]
                ax.plot(epochs,values,color=color)
                ax.set(xlabel='Completed epoch',ylabel=label);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6));ax.grid(True)
                if key=='selection_accuracy':
                    k=t['best_epoch']-1
                    ax.scatter([epochs[k]],[values[k]],marker='*',s=100,color='#D55E00',zorder=4,
                               label=f'Best: epoch {t["best_epoch"]}')
                    ax.set_title(f'Best checkpoint: epoch {t["best_epoch"]}',fontsize=9)
            path=folder/'training_curves.png';fig.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=.12);plt.close(fig)
            (folder/'training_curves.txt').write_text('Training loss, independent checkpoint-selection accuracy, and cosine learning rate. Star marks the validation-selected checkpoint; no test outcomes select it.\n')
            outputs.append(str(path))
    manifest=dict(format='PNG only',dpi=dpi,files=[str(Path(p).relative_to(root)) for p in outputs],
                  seed=cfg['seed'],note='Figures derive from actual saved predictions/metrics. One seed; no across-seed error bars.')
    (root/'figures'/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
