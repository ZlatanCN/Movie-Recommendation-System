import PropTypes from 'prop-types';
import PasswordCriteria from './PasswordCriteria.jsx';

const PasswordStrengthIndicator = (props) => {
  const getStrength = (password) => {
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength += 1;
    if (password.match(/[0-9]/)) strength += 1;
    if (password.match(/[^A-Za-z0-9]/)) strength += 1;
    return strength;
  }

  const convertStrengthToColor = (strength) => {
    switch (strength) {
      case 0: return 'bg-green-500';
      case 1: return 'bg-green-400';
      case 2: return 'bg-yellow-500';
      case 3: return 'bg-yellow-400';
      default: return 'bg-red-500';
    }
  }

  const convertStrengthToText = (strength) => {
    switch (strength) {
      case 0: return 'Very Weak';
      case 1: return 'Weak';
      case 2: return 'Fair';
      case 3: return 'Good';
      default: return 'Strong';
    }
  }

  const strength = getStrength(props.password);

  return (
    <section className={'flex flex-col gap-1'}>
      {/*Password Strength Text*/}
      <section className={'flex justify-between items-center'}>
        <span className={'text-xs text-gray-400'}>
          Password Strength
        </span>
        <span className={'text-xs text-gray-400'}>
          {convertStrengthToText(strength)}
        </span>
      </section>

      {/*Password Strength Level*/}
      <section className={'flex space-x-1 mb-2'}>
        {[...Array(4)].map((_, index) => (
          <div
            key={index}
            className={`h-1 w-1/4 rounded-full transition-colors duration-300 ${index < strength ? convertStrengthToColor(strength) : ('bg-gray-600')}`}
          />
        ))}
      </section>

      {/*Password Criteria*/}
      <PasswordCriteria password={props.password}/>
    </section>
  );
};

PasswordStrengthIndicator.propTypes = {
  password: PropTypes.string.isRequired,
};

export default PasswordStrengthIndicator;