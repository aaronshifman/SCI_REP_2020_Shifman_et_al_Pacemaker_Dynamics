"""Load and manipulate the model into tex or xpp formats."""
import re
from typing import Tuple, List

import brian2 as b2

from model_code import N_FITS, SUPPLEMENTAL_PATH, RECORDING_NAMES, MODEL_PATH
from model_code.data_funcs import get_saved_params

param_ranges = {  # bounds for GA to fit within parameter_name : [bound_min : bound_max]
    'theta_h_inf': [-90, -70] * b2.mV,
    'theta_m_inf': [-70, -50] * b2.mV,
    'theta_n_inf': [-65, -45] * b2.mV,
    'theta_q_inf': [-55, -25] * b2.mV,
    'theta_b_inf': [-70, -60] * b2.mV,
    'theta_g_inf': [-110, -100] * b2.mV,

    'sigma_h_inf': [5, 10] * b2.mV,
    'sigma_m_inf': [5, 10] * b2.mV,
    'sigma_n_inf': [10, 20] * b2.mV,
    'sigma_q_inf': [5, 15] * b2.mV,
    'sigma_b_inf': [10, 20] * b2.mV,
    'sigma_g_inf': [10, 20] * b2.mV,

    'scaling_tau_h': [5, 15] * b2.ms,
    'scaling_tau_m': [0.1, 2] * b2.ms,
    'scaling_tau_n': [5, 15] * b2.ms,
    'scaling_tau_q': [0.1, 2] * b2.ms,
    'scaling_tau_b': [0.1, 2] * b2.ms,
    'scaling_tau_g': [5, 15] * b2.ms,

    'theta_tau_m': [-90, -70] * b2.mV,
    'theta_tau_n': [-65, -45] * b2.mV,
    'theta_tau_q': [-55, -35] * b2.mV,
    'theta_tau_h': [-90, -60] * b2.mV,
    'theta_tau_b': [-100, -80] * b2.mV,
    'theta_tau_g': [-85, -75] * b2.mV,

    'sigma1_tau_h': [5, 15] * b2.mV,
    'sigma2_tau_h': [5, 15] * b2.mV,
    'sigma1_tau_m': [5, 15] * b2.mV,
    'sigma2_tau_m': [5, 15] * b2.mV,
    'sigma1_tau_n': [5, 15] * b2.mV,
    'sigma2_tau_n': [25, 35] * b2.mV,
    'sigma1_tau_q': [10, 20] * b2.mV,
    'sigma2_tau_q': [20, 30] * b2.mV,
    'sigma1_tau_b': [10, 20] * b2.mV,
    'sigma2_tau_b': [10, 20] * b2.mV,
    'sigma1_tau_g': [10, 20] * b2.mV,
    'sigma2_tau_g': [10, 20] * b2.mV,

    'e_na': [20, 30] * b2.mV,
    'e_ca': [20, 30] * b2.mV,
    'e_k': [-90, -80] * b2.mV,
    'e_leak': [-90, -80] * b2.mV,

    'gNa': [30, 70] * b2.msiemens,
    'gK': [30, 70] * b2.msiemens,
    'gLeak': [0, 3] * b2.msiemens,
    'gCaT': [0, 20] * b2.msiemens,
}

"""
Constants for parameters that were free and later fixed by experimentation

Gating conductance m^xy and capacitance (c)
"""
CaT_ex1 = 2
CaT_ex2 = 2
K_ex1 = 2
K_ex2 = 2
Na_ex2 = 1
Na_ex1 = 1

c = 1 * b2.ufarad


def get_generic_model() -> str:
    """Return model (brian) string representation.

    Wrap in a function for brian namespace reasons

    :return: Model
    """
    with open(MODEL_PATH, 'r') as f:
        return ''.join(f.readlines())


def set_model(name: str, num_neurons=N_FITS, params=None) -> Tuple[dict, b2.NeuronGroup]:
    """Create a brian2 model with parameters.

    :param name: Model name to set
    :param num_neurons: Number of neurons to set
    :param params: Optional overwride of parameters if each cell should be unique
    :return: parameters, and model group
    """
    if not params:
        params = get_saved_params(name=name)
    model = get_generic_model()
    model += "I: amp (constant)\n"
    group = b2.NeuronGroup(num_neurons, model, method='euler', dt=0.001 * b2.ms)
    group.v = -50 * b2.mV
    group.I = 0 * b2.amp  # set a control parameter even through not used

    for param_name, param_value in params.items():
        group.__setattr__(param_name, param_value)

    return params, group


def create_xpp_model(name='best_brown_target') -> None:
    """Convert the generic model into an xpp compatible format, and set all free parameters as xpp params.

    This is unequivocally a dumpster fire of misused regex and replacement to get an xpp compatible string.

    While this will not work generally because the intput and output strings are short enough that they can be manually
    checked. Since the input and output strings never change (besides parameter values) this need only be checked
    thoroughly once.

    :param name: Model name to save parameters
    :return: None
    """
    model = get_generic_model()
    params = get_saved_params(name=name)

    # set all exponents to known parameter
    model = model.replace('CaT_ex1', str(CaT_ex1))
    model = model.replace('CaT_ex2', str(CaT_ex2))
    model = model.replace('K_ex1', str(K_ex1))
    model = model.replace('K_ex2', str(K_ex2))
    model = model.replace('Na_ex1', str(Na_ex1))
    model = model.replace('Na_ex2', str(Na_ex2))

    # remove capacitance and bias current (unused) terms
    model = re.sub("(?<=\/)c", str(c.base), model)
    model = re.sub("I[^_]", str(0), model)

    # convert python string representation to xpp representation
    model = model.replace('**', '^')
    model = model.replace('//', '/')

    # add xpp parametters
    param_strings = [f'par {key}={str(val.base)}\n' for key, val in params.items()]

    model_strings = model.split('\n')[1:24]
    model_strings = param_strings + model_strings

    # remove brian unit hints
    for ix, s in enumerate(model_strings):
        bad_ix = s.find(':')
        model_strings[ix] = s[:bad_ix]

    # remove underscores
    lines = '\n'.join(model_strings).replace('_', '')

    # xpp has a problem with scalingtau_x since tau_x is also a variable
    lines = re.sub("(scalingtau)(?=(.))", 'scalet', lines)

    # finally save data
    with open(f'data/figure_data/{name}.ode', 'w') as f:
        f.writelines(lines)


def latex_convert(m: str) -> str:
    """Convert the generic model into an LaTeX format, and save equations / tables.

    Again this is unequivocally a dumpster fire of misused regex and replacement to get an xpp compatible string.

    While this will not work generally because the intput and output strings are short enough that they can be manually
    checked. Since the input and output strings never change (besides parameter values) this need only be checked
    thoroughly once.

    :param m: Model string
    :return: Converted model string
    """
    m = m.replace('CaT_ex1', str(CaT_ex1))  # set exponent to constant
    m = m.replace('CaT_ex2', str(CaT_ex2))  # set exponent to constant
    m = m.replace('K_ex1', str(K_ex1))  # set exponent to constant
    m = m.replace('K_ex2', str(K_ex2))  # set exponent to constant
    m = m.replace('Na_ex1', str(Na_ex1))  # set exponent to constant
    m = m.replace('Na_ex2', str(Na_ex2))  # set exponent to constant

    m = re.sub("/c", "", m)  # remove capacitance as = 1
    m = re.sub('\((?=I)', "",
               m)  # remove "(" with succeding "I" -- represents open parenthesis in current balance - not needed
    m = re.sub('(?<=K)\)', "",
               m)  # Remove ")" with preceding "K"-- represents close parenthesis in current balance - not needed
    m = re.sub('I(?!_).*?\+', "",
               m)  # Remove "I" without subsequent "_" and its subsequent + -- remove "I +" which remove bias current from equation
    m = m.replace('**', '^')  # Convert python expoenentiation to latex
    m = m.replace('//1', '')  # Remove float -> int conversion
    m = m.replace('(2)', '2')  # convert ^(2) -> ^2
    m = m.replace('^(1)', '')  # remove ^1 exponentiation
    m = m.replace('_inf', '_{inf}')  # Encapsulate inf into subscript
    m = re.sub("(?<=sigma)[1-9]", lambda x: "^" + x.group(),
               m)  # find "1" or "2" with preceding "sigma" and convert them to superscript for notation
    m = re.sub("(?<=theta_)._{inf}", lambda x: "{" + x.group() + "}",
               m)  # Convert theta_a_inf to theta_{a_inf} where "a" is any character for better rendering
    m = re.sub("(?<=sigma_)._{inf}", lambda x: "{" + x.group() + "}",
               m)  # Same as above but for sigma_a_inf -> sigma_{a_inf}
    m = re.sub("tau_.", lambda x: "{" + x.group() + "}", m)  # "tau_a" is typically a subscript so encapsulate incase
    m = re.sub(':(.*)', "", m)  # Remove brian trailing unit notation i.e. ": volt"
    m = m.replace('e_na', 'E_{Na}')  # Fix e_na notation
    m = m.replace('e_k', 'E_{K}')  # Fix e_k notation
    m = m.replace('e_ca', 'E_{Ca}')  # Fix e_ca notation
    m = m.replace('e_leak', 'E_{Leak}')  # Fix e_leak notation
    m = m.replace('gNa', 'G_{Na}')  # Fix g_na notation
    m = m.replace('gK', 'G_K')  # Fix g_k notation
    m = m.replace('gCaT', 'G_{Ca}')  # Fix g_ca notation
    m = m.replace('gLeak', 'G_{Leak}')  # Fix g_leak notation
    m = m.replace('inf', '\\infty')  # convert inf -> \infty for rendering
    m = m.replace('tau', '\\tau')  # convert tau -> \tau for rendering
    m = m.replace('I_leak', 'I_{Leak}')  # Encapsulate Leak ion subscript
    m = m.replace('I_Ca', 'I_{Ca}')  # Encapsulate Ca ion subscript
    m = m.replace('I_Na', 'I_{Na}')  # Encapsulate Na ion subscript

    # da/dt styling
    m = re.sub('d./dt', lambda x: "\\frac{" + x.group().split('/')[0] + "}{" + x.group().split('/')[1] + "}",
               m)  # convert "da/dt" to \\frac{da}{dt} for rendering
    m = re.sub('(\(. - .*\))', lambda x: '\\frac{' + x.group() + "}",
               m)  # encapusulate a-a_inf from da/dt into \frac{a-a_inf}
    m = re.sub('/(?=\{\\\\tau_.\})', '',
               m)  # remove division symbol between "a-a_inf" and "tau_a" in da/dt. tau_a is encapsulated above

    m = re.sub("\([qnmhgb]\^?[1-9]?\)", lambda x: x.group().split('(')[1].split(')')[0],
               m)  # remove redundant parentheses around a^2 (or any gating variable)
    m = re.sub("scaling_.*\/(?=\(exp\()", lambda x: "\\frac{" + x.group()[:-1] + "}",
               m)  # encapsulate, begin fraction and remove trailing division numerator of equations with scaling (tau_a)
    m = re.sub("\(exp.*", lambda x: "{" + x.group() + "}", m)  # encapsulate denominator of last step
    m = re.sub('\(theta.*?- v\)\/', lambda x: "\\frac{" + x.group()[:-1] + "}",
               m)  # in global denominator of last step convert new numerator to fraction as before
    m = re.sub('sigma.*?\}', lambda x: "{" + x.group() + "}", m)  # same as above for denominator
    m = m.replace('theta', '\\theta')  # stylize theta
    m = m.replace('sigma', '\\sigma')  # stylize sigma
    # m = re.sub('(?<=._{\\\\infty\} = ){.*', lambda x: '{' + x.group() + "}", m)
    m = re.sub('(?<=._{\\\\infty\} = )1\/', lambda x: "\\frac{1}",
               m)  # in a_inf equation encapsulate the "1/" as a fraction
    m = re.sub('\(\\\\theta.*?-.*?v\)', lambda x: x.group()[1:-1],
               m)  # since theta_a - v is now in a division remove leading and trailing "(" and ")"
    m = m.replace('exp', '\\exp')  # stylize exp
    m = re.sub(r'\((?=\\exp)', '', m)  # remove leading "(" from exp as its now in a denominator
    m = re.sub('(?<=[1\)])\)(?= )', '',
               m)  # Remove trailing ")" from denomiators as the pattern is either "+1)" or "))"
    m = re.sub('(?<=}})\)', r'\\right )', m)  # Convert trailing ")" to \right )
    m = re.sub('(?<=exp)\(', '\\left (', m)  # Convert leading "(" to \left (
    # m = m.replace('*', '\\times ')
    m = m.replace('*', '')  # Remove * opperator
    m = re.sub("(?<={)\(.*?\)", lambda x: x.group()[1:-1],
               m)  # Remove leading and trailing "(" and ")" from "a-a_inf" terms
    m = re.sub(r"(?<=\\frac{). - .*?}", lambda x: '-'.join(x.group().split('-')[::-1]),
               m)  # flip order of a-a_inf terms
    m = re.sub(r"(?<=\= )-(?=\\frac)", '', m)  # Remove leading "-" in a_inf-a as its been flipped above
    m = re.sub("E_.*?v", lambda x: '-'.join(x.group().split('-')[::-1]), m)  # convert E_a - v to v-E_a for convention
    m = m.replace('scaling', 's')  # Convert scaling
    m = re.sub('\+(?= I)', '-', m)  # Convert I_na + ... -> I_na - ...
    m = re.sub('(?<==..)I_{Leak}', '-I_{Leak}', m)  # Flip sign of I_{Leak} when it matches the dv/dt pattern
    return m


def write_equations(model_strings: List[str]) -> None:
    """Write model into equations.

    :param model_strings: Converted LaTeX model
    :return: None
    """
    with open(SUPPLEMENTAL_PATH / 'equations.tex', 'w') as f:
        for ix, m in enumerate(model_strings):
            st = f"{m}\\\\\n".replace('=', '&=')
            if ix + 1 == len(model_strings):
                st = st[:-3]
            f.write(st)


def write_fit_tables() -> None:
    """Write a table for gating variables and ionic parameters.

    :return: None
    """
    params = get_saved_params(name=f'best_{RECORDING_NAMES[0]}')
    tables = ["", ""]  # g/E, sigma/theta
    for param, val in sorted(params.items(), key=lambda x: x[0]):
        cvt_param = latex_convert(param)
        if cvt_param[0] == "{":
            cvt_param = cvt_param[1:-1]
        ix = 0 if (param.startswith('g') or param.startswith('e_')) else 1

        out_str = "$" + cvt_param + "$ "

        for n in RECORDING_NAMES:
            out_val = get_saved_params(name=f'best_{n}')[param]
            out_str += "&" + '{0:.2f}'.format(float(out_val.in_best_unit(2).split(' ')[0]))

        out_str += "&" + out_val.in_best_unit().split(' ')[
            1] + '\\\\\n'
        tables[ix] += out_str

    with open(SUPPLEMENTAL_PATH / 'param_current.tex', 'w') as f:
        f.writelines(tables[0])

    with open(SUPPLEMENTAL_PATH / 'param_gating.tex', 'w') as f:
        f.writelines(tables[1])


def write_bounds_table() -> None:
    """Write a list of parameter fitting bounds to a table.

    :return: None
    """
    table = ""
    for param, val in sorted(param_ranges.items(), key=lambda x: x[0]):
        cvt_param = latex_convert(param)
        if cvt_param[0] == "{":
            cvt_param = cvt_param[1:-1]

        table += "$" + cvt_param + "$ " + "&" + val[0].in_best_unit(2).split(' ')[0] + "&" + \
                 val[1].in_best_unit().split(' ')[
                     0] + "&" + val[1].in_best_unit().split(' ')[
                     1] + '\\\\\n'
    with open(SUPPLEMENTAL_PATH / 'bounds.tex', 'w') as f:
        f.writelines(table)


def create_data_table() -> None:
    """Create equations and table for fit parameters and bounds.

    :return: None
    """
    model = get_generic_model()
    m = latex_convert(model)
    model_strings = m.split('\n')[0:23]  # take only equations not parameter declarations

    write_equations(model_strings)
    write_fit_tables()
    write_bounds_table()
