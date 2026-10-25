import numpy as np
import pytest
from scipy.stats import binom
from shiftguard.risk_control import upper_binomial, candidate_grids, certify, evaluate_certificates


def test_zero_failure_bound_and_edges():
    assert upper_binomial(0,100,.01) == pytest.approx(1-.01**.01)
    assert upper_binomial(0,0,.05) == 1
    assert upper_binomial(10,10,.05) == 1
    with pytest.raises(ValueError):
        upper_binomial(3,2,.05)


def test_exact_binomial_noncoverage_probability():
    # Independently integrate the binomial sampling distribution, not a simulation.
    n,delta=40,.05
    for p in [.01,.05,.1,.3,.7,.95]:
        bad=[k for k in range(n+1) if upper_binomial(k,n,delta) < p]
        probability=sum(binom.pmf(k,n,p) for k in bad)
        assert probability <= delta+1e-12


def test_simultaneous_family_and_whole_tie_acceptance():
    grids=candidate_grids({'a':np.array([0.,0.,1.,1.]),'b':np.array([0.,1.,2.,3.])},3)
    scores={'a':np.r_[np.zeros(100),np.ones(100)],'b':np.r_[np.zeros(100),np.ones(100)*3]}
    errors=np.zeros(200,dtype=int)
    certificate=certify(scores,errors,grids,[100,200],[.05,.1],.05,20)
    assert certificate['family_size'] == 2*sum(len(g) for g in grids.values())
    assert certificate['per_rule_delta'] == pytest.approx(.05/certificate['family_size'])
    assert len(certificate['selections']) == 2*2*2
    assert all(r['accepted'] in [0,100,200] for r in certificate['candidate_bounds'])
    assert any(r['status']=='certified' for r in certificate['selections'])


def test_no_certificate_is_not_zero_risk_and_test_labels_cannot_change_threshold():
    scores={'a':np.arange(100,dtype=float)}
    c=certify(scores,np.ones(100,dtype=int),{'a':[30.,90.]},[100],[.1])
    frozen=c['selections'][0]['threshold']
    rows=evaluate_certificates(c,scores,np.zeros(100))
    assert rows[0]['test_coverage'] == 0 and rows[0]['test_risk'] is None
    assert c['selections'][0]['threshold'] is frozen is None
    with pytest.raises(ValueError):
        certify(scores,np.zeros(100),{'a':[30.]},[101])
