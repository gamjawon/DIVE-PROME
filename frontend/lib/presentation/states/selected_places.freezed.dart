// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'selected_places.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$SelectedPlaces {

 Location? get start; Location? get end;
/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$SelectedPlacesCopyWith<SelectedPlaces> get copyWith => _$SelectedPlacesCopyWithImpl<SelectedPlaces>(this as SelectedPlaces, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is SelectedPlaces&&(identical(other.start, start) || other.start == start)&&(identical(other.end, end) || other.end == end));
}


@override
int get hashCode => Object.hash(runtimeType,start,end);

@override
String toString() {
  return 'SelectedPlaces(start: $start, end: $end)';
}


}

/// @nodoc
abstract mixin class $SelectedPlacesCopyWith<$Res>  {
  factory $SelectedPlacesCopyWith(SelectedPlaces value, $Res Function(SelectedPlaces) _then) = _$SelectedPlacesCopyWithImpl;
@useResult
$Res call({
 Location? start, Location? end
});


$LocationCopyWith<$Res>? get start;$LocationCopyWith<$Res>? get end;

}
/// @nodoc
class _$SelectedPlacesCopyWithImpl<$Res>
    implements $SelectedPlacesCopyWith<$Res> {
  _$SelectedPlacesCopyWithImpl(this._self, this._then);

  final SelectedPlaces _self;
  final $Res Function(SelectedPlaces) _then;

/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? start = freezed,Object? end = freezed,}) {
  return _then(_self.copyWith(
start: freezed == start ? _self.start : start // ignore: cast_nullable_to_non_nullable
as Location?,end: freezed == end ? _self.end : end // ignore: cast_nullable_to_non_nullable
as Location?,
  ));
}
/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get start {
    if (_self.start == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.start!, (value) {
    return _then(_self.copyWith(start: value));
  });
}/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get end {
    if (_self.end == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.end!, (value) {
    return _then(_self.copyWith(end: value));
  });
}
}


/// Adds pattern-matching-related methods to [SelectedPlaces].
extension SelectedPlacesPatterns on SelectedPlaces {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _SelectedPlaces value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _SelectedPlaces() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _SelectedPlaces value)  $default,){
final _that = this;
switch (_that) {
case _SelectedPlaces():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _SelectedPlaces value)?  $default,){
final _that = this;
switch (_that) {
case _SelectedPlaces() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( Location? start,  Location? end)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _SelectedPlaces() when $default != null:
return $default(_that.start,_that.end);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( Location? start,  Location? end)  $default,) {final _that = this;
switch (_that) {
case _SelectedPlaces():
return $default(_that.start,_that.end);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( Location? start,  Location? end)?  $default,) {final _that = this;
switch (_that) {
case _SelectedPlaces() when $default != null:
return $default(_that.start,_that.end);case _:
  return null;

}
}

}

/// @nodoc


class _SelectedPlaces implements SelectedPlaces {
  const _SelectedPlaces({this.start, this.end});
  

@override final  Location? start;
@override final  Location? end;

/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$SelectedPlacesCopyWith<_SelectedPlaces> get copyWith => __$SelectedPlacesCopyWithImpl<_SelectedPlaces>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _SelectedPlaces&&(identical(other.start, start) || other.start == start)&&(identical(other.end, end) || other.end == end));
}


@override
int get hashCode => Object.hash(runtimeType,start,end);

@override
String toString() {
  return 'SelectedPlaces(start: $start, end: $end)';
}


}

/// @nodoc
abstract mixin class _$SelectedPlacesCopyWith<$Res> implements $SelectedPlacesCopyWith<$Res> {
  factory _$SelectedPlacesCopyWith(_SelectedPlaces value, $Res Function(_SelectedPlaces) _then) = __$SelectedPlacesCopyWithImpl;
@override @useResult
$Res call({
 Location? start, Location? end
});


@override $LocationCopyWith<$Res>? get start;@override $LocationCopyWith<$Res>? get end;

}
/// @nodoc
class __$SelectedPlacesCopyWithImpl<$Res>
    implements _$SelectedPlacesCopyWith<$Res> {
  __$SelectedPlacesCopyWithImpl(this._self, this._then);

  final _SelectedPlaces _self;
  final $Res Function(_SelectedPlaces) _then;

/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? start = freezed,Object? end = freezed,}) {
  return _then(_SelectedPlaces(
start: freezed == start ? _self.start : start // ignore: cast_nullable_to_non_nullable
as Location?,end: freezed == end ? _self.end : end // ignore: cast_nullable_to_non_nullable
as Location?,
  ));
}

/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get start {
    if (_self.start == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.start!, (value) {
    return _then(_self.copyWith(start: value));
  });
}/// Create a copy of SelectedPlaces
/// with the given fields replaced by the non-null parameter values.
@override
@pragma('vm:prefer-inline')
$LocationCopyWith<$Res>? get end {
    if (_self.end == null) {
    return null;
  }

  return $LocationCopyWith<$Res>(_self.end!, (value) {
    return _then(_self.copyWith(end: value));
  });
}
}

// dart format on
