import PropTypes from 'prop-types';
import { CheckOutlined, CloseOutlined } from '@ant-design/icons';

const PasswordCriteria = (props) => {
  const criteria = [
    {label: 'At least 8 characters', isValid: props.password.length >= 8},
    {label: 'At least 1 uppercase letter', isValid: /[A-Z]/.test(props.password)},
    {label: 'At least 1 lowercase letter', isValid: /[a-z]/.test(props.password)},
    {label: 'At least 1 number', isValid: /[0-9]/.test(props.password)},
    {label: 'At least 1 special character', isValid: /[^A-Za-z0-9]/.test(props.password)},
  ]

  return (
    <div className={'space-y-1'}>
      {criteria.map((item, index) => (
        <div key={item.label} className={'flex items-center text-xs'}>
          {item.isValid ? (
            <CheckOutlined className={'text-red-600 pr-1'}/>
          ) : (
            <CloseOutlined className={'text-red-600 pr-1'}/>
          )}
          <span className={item.isValid ? ('text-red-600') : ('text-gray-400')}>
            {item.label}
          </span>
        </div>
      ))}
    </div>
  );
};

PasswordCriteria.propTypes = {
  password: PropTypes.string.isRequired,
};

export default PasswordCriteria;